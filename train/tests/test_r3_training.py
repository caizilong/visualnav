"""Numerical contract tests to run in the Ubuntu training environment."""

import copy
import random
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from diffusers.training_utils import EMAModel
from vint_train.data.simulation import digest_json

from vint_train.models.action_codec import normalize_actions, decode_actions
from vint_train.models.nomad.goal_relation_head import GoalRelationHead
from vint_train.training.goal_relation_train import (
    window_denominators, weighted_loss, optimizer_update, relation_weight,
)
from vint_train.training.simulation_checkpoint import make_sim_checkpoint, restore_sim_checkpoint
from vint_train.training.simulation_train import isolated_evaluation_rng


def action_batch(valid):
    return (torch.zeros(len(valid), 3, 1, 1), None, None, None, None, None, torch.tensor(valid))


class WindowLossTests(unittest.TestCase):
    def test_uneven_valid_counts_and_tail_gradient_equal_concatenation(self):
        a = [action_batch([1, 0, 1]), action_batch([1, 0])]
        masks = [torch.tensor([0, 1, 0]), torch.tensor([1, 0])]
        r = [{"reach_valid": torch.tensor([True, False, True]), "distance_valid": torch.tensor([True, False, False])},
             {"reach_valid": torch.tensor([False, True]), "distance_valid": torch.tensor([False, True])}]
        denom = window_denominators(a, masks, r)
        self.assertAlmostEqual(denom["action"], 3.05)
        self.assertAlmostEqual(denom["temporal"], 3.05)
        self.assertEqual((denom["reach"], denom["distance"]), (3, 2))
        prediction = torch.tensor([.1, .2, -.3, .4, .5], requires_grad=True)
        target = torch.tensor([1., 0., 1., 0., 0.])
        reaches = torch.cat([batch["reach_valid"] for batch in r])
        distances = torch.cat([batch["distance_valid"] for batch in r])
        weights = {"action": .9999, "temporal": .0001, "reach": .4, "distance": .4}
        expected = (.9999 * prediction[[0, 2, 3]].square().sum() / 3.05
                    + .0001 * (prediction[[0, 2, 4]] - 1).square().sum() / 3.05
                    + .4 * F.binary_cross_entropy_with_logits(prediction[reaches], target[reaches])
                    + .4 * F.smooth_l1_loss(prediction[distances], target[distances], beta=.1))
        reference_grad = torch.autograd.grad(expected, prediction)[0]
        cursor = 0
        for batch, mask, relation in zip(a, masks, r):
            stop = cursor + len(batch[0])
            p, y = prediction[cursor:stop], target[cursor:stop]
            sums = {"action": p[batch[6] > 0].square().sum(),
                    "temporal": (p[mask == 0] - 1).square().sum(),
                    "reach": F.binary_cross_entropy_with_logits(p[relation["reach_valid"]], y[relation["reach_valid"]], reduction="sum"),
                    "distance": F.smooth_l1_loss(p[relation["distance_valid"]], y[relation["distance_valid"]], beta=.1, reduction="sum")}
            weighted_loss(sums, denom, weights).backward()
            cursor = stop
        torch.testing.assert_close(prediction.grad, reference_grad)

    def test_all_unknown_has_no_optimizer_objective(self):
        den = window_denominators([action_batch([0])], [torch.ones(1)],
                                  [{"reach_valid": torch.zeros(1, dtype=torch.bool), "distance_valid": torch.zeros(1, dtype=torch.bool)}])
        zero = {key: torch.tensor(0.) for key in ("action", "temporal", "reach", "distance")}
        self.assertIsNone(weighted_loss(zero, den, {key: 1. for key in zero}))
        self.assertEqual(relation_weight(0), .1)
        self.assertEqual(relation_weight(500), 1.)

    def test_physical_action_roundtrip_and_positive_head(self):
        actions = torch.tensor([[[.3, -.1], [.7, .2], [1., .1]]])
        stats = {"min": [-1., -1.], "max": [2., 2.]}
        torch.testing.assert_close(decode_actions(normalize_actions(actions, stats), stats, .7), actions * .7)
        head = GoalRelationHead(256, 3.)
        result = head(torch.randn(2, 256))
        self.assertEqual(result["reachability_logit"].shape, (2,))
        self.assertTrue((result["distance_m"] > 0).all())
        (result["reachability_logit"].sum() + result["distance_m"].sum()).backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters()))


class FakeStandardScaler:
    """Emulate GradScaler's step=None / scale backoff contract without CUDA."""
    def __init__(self, skip):
        self.skip, self.scale_value = skip, 1024.

    def get_scale(self):
        return self.scale_value

    def is_enabled(self):
        return True

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        if not self.skip:
            optimizer.step()
        return None

    def update(self):
        if self.skip:
            self.scale_value *= .5


class UpdateAndResumeTests(unittest.TestCase):
    def test_step_none_is_success_and_backoff_is_skip(self):
        for skip in (False, True):
            model = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            before = model.weight.detach().clone()
            model(torch.ones(1, 1)).sum().backward()
            updated, _ = optimizer_update(model, optimizer, FakeStandardScaler(skip), 1.)
            self.assertEqual(updated, not skip)
            self.assertEqual(torch.equal(before, model.weight), skip)

    def setup_run(self):
        model = torch.nn.Linear(2, 1)
        model._model_config = {"model_type": "nomad", "test_model": "linear"}
        optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "name": "all"}], lr=.001)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.1, total_iters=2),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)], milestones=[2])
        ema = EMAModel(model=model, power=.75)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        return model, optimizer, scheduler, ema, scaler

    def step_run(self, run):
        model, optimizer, scheduler, ema, scaler = run
        optimizer.zero_grad(set_to_none=True)
        model(torch.tensor([[.2, .4]])).square().sum().backward()
        updated, _ = optimizer_update(model, optimizer, scaler, 1.)
        self.assertTrue(updated)
        ema.step(model)
        scheduler.step()

    def test_epoch_resume_restores_next_update_and_rejects_scale_change(self):
        run = self.setup_run()
        self.step_run(run)
        model, optimizer, scheduler, ema, scaler = run
        spec = {"action_stats": {"min": [-1., -1.], "max": [1., 1.]},
                "action_scale_m": .5, "action_time_step_s": .2, "observation_stride": 1}
        data_manifest = {"manifest": {}, "files": {}}
        fingerprint = digest_json(data_manifest)
        saved = copy.deepcopy(make_sim_checkpoint(model, ema, optimizer, scheduler, scaler, 0, 1, spec, fingerprint, spec, data_manifest))
        expected_rng = (random.random(), float(np.random.rand()), torch.rand(1))
        self.step_run(run)
        restored = self.setup_run()
        other_model, other_optimizer, other_scheduler, other_ema, other_scaler = restored
        epoch, steps = restore_sim_checkpoint(saved, other_model, other_ema, other_optimizer, other_scheduler, other_scaler, spec, fingerprint)
        self.assertEqual((epoch, steps), (1, 1))
        self.assertEqual((random.random(), float(np.random.rand())), expected_rng[:2])
        torch.testing.assert_close(torch.rand(1), expected_rng[2])
        self.step_run(restored)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, other_model.state_dict()[key])
            torch.testing.assert_close(ema.averaged_model.state_dict()[key], other_ema.averaged_model.state_dict()[key])
        self.assertEqual(ema.optimization_step, other_ema.optimization_step)
        self.assertEqual(optimizer.param_groups[0]["lr"], other_optimizer.param_groups[0]["lr"])
        bad = dict(spec, action_scale_m=1.)
        with self.assertRaises(ValueError):
            restore_sim_checkpoint(saved, other_model, other_ema, other_optimizer, other_scheduler, other_scaler, bad, fingerprint)

    def test_evaluation_does_not_consume_rng(self):
        random.seed(12)
        np.random.seed(12)
        torch.manual_seed(12)
        expected = (random.random(), float(np.random.rand()), torch.rand(1))
        random.seed(12)
        np.random.seed(12)
        torch.manual_seed(12)
        with isolated_evaluation_rng([]):
            random.random()
            np.random.rand()
            torch.rand(1)
        self.assertEqual((random.random(), float(np.random.rand())), expected[:2])
        torch.testing.assert_close(torch.rand(1), expected[2])


if __name__ == "__main__":
    unittest.main()
