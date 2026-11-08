"""Regression/integration checks to run in the configured Ubuntu environment.

These tests have not been executed in the Mac editing session.
"""

import copy
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from test_r1_core import definition
from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.data.trajectory_regions import fit_settings, fit_visitation_prior
from vint_train.models.action_codec import normalize_actions, decode_actions
from vint_train.models.factory import build_model
from vint_train.models.weights import weights_sha256
from vint_train.models.nomad.spatial_bottleneck import SpatialBottleneck, GlobalVisitationHead
from vint_train.training.trajectory_visitation import visitation_targets, visitation_loss_sum, VisitationMetrics
from vint_train.training.experiment_protocol import (DiffusionStreams, branch_rng,
    audit_scratch_initialization, EXTENSION_PREFIXES)
from vint_train.training.compare_ablation_runs import compare_runs
from vint_train.inference.calibration import calibration_binding, load_calibration


class CoordinateContractTest(unittest.TestCase):
    def test_dataset_rotation_stride_scale_and_diffusion_roundtrip(self):
        # Exercise the actual dataset action method without opening an image cache.
        dataset = object.__new__(ViNT_Dataset)
        dataset.len_traj_pred, dataset.waypoint_spacing = 4, 2
        dataset.learn_angle, dataset.normalize, dataset.num_action_params = False, True, 2
        dataset.data_config = {"metric_waypoint_spacing": 1.25}  # scale = 1.25 * 2
        local = np.array([[.5, 0], [.75, 1.3], [-2.5, 0], [2.25, -3.9]])
        yaw, origin, now = math.pi / 3, np.array([10., 20.]), 2
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        positions = np.repeat(origin[None], 11, axis=0)
        positions[now + np.arange(1, 5) * 2] = local @ rotation.T + origin
        trajectory = {"position": positions, "yaw": np.full(11, yaw)}
        actions, _ = dataset._compute_actions(trajectory, now, now + 4)
        points = torch.tensor(actions, dtype=torch.float32)[None]
        config = {"action_scale_m": 2.5, "r1": {"discretization": definition()}}
        batch = (None, None, points, None, None, torch.zeros(1, dtype=torch.long), torch.ones(1))
        target, valid, counts = visitation_targets(batch, config)
        expected = torch.zeros(1, 6, 4)
        expected[0, 0, 0] = expected[0, 1, 1] = expected[0, 3, 2] = expected[0, 5, 3] = .25
        torch.testing.assert_close(target, expected)
        self.assertEqual((valid.tolist(), counts.tolist()), ([True], [4]))
        torch.testing.assert_close(points[0] * 2.5, torch.tensor(local, dtype=torch.float32))
        stats = {"min": [-3., -3.], "max": [3., 3.]}
        restored = decode_actions(normalize_actions(points, stats), stats, 2.5)
        torch.testing.assert_close(restored[0], torch.tensor(local, dtype=torch.float32), atol=2.e-6, rtol=1.e-5)
        # An action-invalid row must not contaminate targets, even with NaN padding.
        batch = (None, None, torch.cat((points, torch.full_like(points, float("nan")))), None, None,
                 torch.zeros(2, dtype=torch.long), torch.tensor([1., 0.]))
        target, valid, counts = visitation_targets(batch, config)
        self.assertEqual(valid.tolist(), [True, False])
        self.assertEqual(float(target[1].sum()), 0.)

    def test_unsupported_dataset_scale_and_bad_scale_are_rejected(self):
        batch = (None, None, torch.ones(1, 4, 2), None, None, torch.ones(1), torch.ones(1))
        config = {"action_scale_m": 2., "r1": {"discretization": definition()}}
        with self.assertRaisesRegex(ValueError, "single-manifest"):
            visitation_targets(batch, config)
        config["action_scale_m"] = 0.
        with self.assertRaisesRegex(ValueError, "positive"):
            visitation_targets(batch, config)

    def test_prior_uses_valid_sample_mass_not_waypoint_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "episode").mkdir()
            with (root / "episode/traj_data.pkl").open("wb") as stream:
                pickle.dump({"position": np.array([[0, 0], [0, 0], [1, 0], [2, 1], [3, 2]], dtype=float),
                             "yaw": np.zeros(5)}, stream)
            bundle = {"root": root, "trajectories": {"train": [{"name": "episode"}],
                      "validation": [{"name": "must-not-be-read"}]},
                      "manifest": {"waypoint_spacing": 1, "action_scale_m": 2.}}
            config = {"context_size": 0, "len_traj_pred": 2, "distance": {"max_dist_cat": 3},
                      "action": {"min_dist_cat": 0, "max_dist_cat": 4}, "r1": {}}
            prior = fit_visitation_prior(bundle, config, definition(), fit_settings(config))
            expected = np.zeros((6, 4))
            expected[0, 1], expected[0, 2] = .75 + .75 / 2, .75 / 2
            expected[1, 1] = expected[1, 2] = (2 / 3) / 2
            expected /= 1.5 + 2 / 3
            expected = (1 - 1.e-4) * expected + 1.e-4 / 24
            np.testing.assert_allclose(np.array(prior["probability_flat"]).reshape(6, 4), expected)
            self.assertEqual(prior["valid_windows"], 3)


class ControlsAndRandomnessTest(unittest.TestCase):
    def test_pooled_and_global_controls_and_auxiliary_gradient_routes(self):
        for representation in ("query_tokens", "pooled_spatial"):
            module = SpatialBottleneck(12, 16, token_dim=8, num_attention_heads=2, representation=representation)
            features = torch.randn(2, 12, 3, 4, requires_grad=True)
            condition = torch.randn(2, 16, requires_grad=True)
            tokens = module.encode(features)
            self.assertEqual(tuple(tokens.shape), (2, 6, 8))
            torch.testing.assert_close(module.fuse(condition, tokens), condition, atol=0, rtol=0)
            q = torch.zeros(2, 6, 4)
            q[:, 1, 2] = 1.
            visitation_loss_sum(module.predict_visitation(tokens, condition), q, torch.ones(2, dtype=torch.bool)).backward()
            self.assertGreater(float(features.grad.norm()), 0.)
            self.assertGreater(float(condition.grad.norm()), 0.)
            self.assertIsNone(module.residual_projection.weight.grad)
            self.assertEqual(hasattr(module, "queries"), representation == "query_tokens")
        head = GlobalVisitationHead(16, token_dim=8)
        condition = torch.randn(2, 16, requires_grad=True)
        logits = head.predict_visitation(None, condition)
        self.assertEqual(tuple(logits.shape), (2, 6, 4))
        logits.square().sum().backward()
        self.assertGreater(float(condition.grad.norm()), 0.)
        self.assertFalse(hasattr(head, "spatial_projection"))

    def test_diffusion_streams_ignore_global_and_r3_rng_consumption(self):
        cpu = torch.device("cpu")
        a, b = DiffusionStreams(5, 3, cpu), DiffusionStreams(5, 3, cpu)
        targets = torch.zeros(3, 4, 2)
        for _ in range(3):
            expected = a.draw(targets, 10)
            torch.randn(123)
            with branch_rng(29, cpu):
                torch.randn(1000)  # Stands for a relation-branch stochastic forward.
            actual = b.draw(targets, 10)
            for x, y in zip(expected, actual):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
        fresh_epoch, resumed_epoch = DiffusionStreams(5, 4, cpu), DiffusionStreams(5, 4, cpu)
        for x, y in zip(fresh_epoch.draw(targets, 10), resumed_epoch.draw(targets, 10)):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        before = torch.get_rng_state().clone()
        with branch_rng(17, cpu):
            first = torch.rand(20)
        with branch_rng(51, cpu):
            torch.rand(39)
        with branch_rng(17, cpu):
            second = torch.rand(20)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_token_permutation_detects_reliance_and_skips_singletons(self):
        class TokenOnlyPredictor:
            def predict_visitation(self, tokens, condition):
                return tokens + condition[:, None, :]
        model = SimpleNamespace(training=False, vision_encoder=SimpleNamespace(
            spatial_bottleneck=TokenOnlyPredictor(), r1_mode="auxiliary"))
        logits = torch.full((2, 6, 4), -8.)
        logits[0, 0, 0] = logits[1, 3, 1] = 8.
        q = (logits == 8.).float()
        encoded = {"spatial_tokens": logits, "visitation_logits": logits, "base_condition": torch.zeros(2, 4)}
        metrics = VisitationMetrics(prior_probability=[1 / 24] * 24, token_permutation=True)
        valid, count = torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.long)
        metrics.update(logits, q, valid, count)
        metrics.update_token_permutation(model, encoded, q, valid, count)
        report = metrics.report()
        self.assertGreater(report["token_permutation"]["cross_entropy_increase"], 10.)
        self.assertAlmostEqual(report["train_prior"]["cross_entropy"], math.log(24), places=5)
        metrics.update_token_permutation(model, {k: v[:1] for k, v in encoded.items()}, q[:1], valid[:1], count[:1])
        self.assertEqual(metrics.report()["token_permutation"]["skipped_singleton_samples"], 1)


class InitializationAndCalibrationTest(unittest.TestCase):
    def test_current_frame_tokens_goal_mask_and_identity_feedback(self):
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/nomad_r1_r3_habitat.yaml").read_text())
        config["r1"]["discretization"] = definition()
        config["r3"]["enabled"] = False
        torch.manual_seed(3)
        model = build_model(config).eval()
        images, goal = torch.randn(2, 12, 96, 96), torch.randn(2, 3, 96, 96)
        mask = torch.ones(2, dtype=torch.long)
        with torch.no_grad():
            first = model("vision_encoder", obs_img=images, goal_img=goal, input_goal_mask=mask, return_r1=True)
            other_goal = model("vision_encoder", obs_img=images, goal_img=torch.randn_like(goal),
                               input_goal_mask=mask, return_r1=True)
            history_changed = images.clone()
            history_changed[:, :-3] = torch.randn_like(history_changed[:, :-3])
            other_history = model("vision_encoder", obs_img=history_changed, goal_img=goal,
                                  input_goal_mask=mask, return_r1=True)
            independent = model("spatial_tokens", obs_img=images)
        torch.testing.assert_close(first["condition"], first["base_condition"], atol=0, rtol=0)
        torch.testing.assert_close(first["condition"], other_goal["condition"], atol=1.e-5, rtol=1.e-4)
        torch.testing.assert_close(first["spatial_tokens"], other_history["spatial_tokens"], atol=1.e-5, rtol=1.e-4)
        torch.testing.assert_close(first["spatial_tokens"], independent, atol=1.e-5, rtol=1.e-4)

    def test_factory_common_weights_and_extension_initialization_match(self):
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/nomad_r1_r3_habitat.yaml").read_text())
        config["seed"] = 11
        config["r1"]["discretization"] = definition()
        config["r3"]["label_definition"] = {"max_path_length_m": 20.}
        fingerprints = {}
        for r1, r3 in ((False, False), (True, False), (False, True), (True, True)):
            config["r1"]["enabled"], config["r3"]["enabled"] = r1, r3
            torch.manual_seed(config["seed"])
            model = build_model(config)
            fingerprints[r1, r3] = audit_scratch_initialization(model, config)
            del model
        self.assertEqual(len({x["initial_common_weights_sha256"] for x in fingerprints.values()}), 1)
        self.assertEqual(fingerprints[True, False]["initial_r1_weights_sha256"], fingerprints[True, True]["initial_r1_weights_sha256"])
        self.assertEqual(fingerprints[False, True]["initial_r3_weights_sha256"], fingerprints[True, True]["initial_r3_weights_sha256"])
        for variant in ("pooled_spatial", "global_condition"):
            config["r1"].update(representation=variant, mode="auxiliary")
            torch.manual_seed(config["seed"])
            model = build_model(config)
            self.assertEqual(weights_sha256(model, exclude_prefixes=EXTENSION_PREFIXES),
                             fingerprints[False, False]["initial_common_weights_sha256"])
            if variant == "global_condition":
                with torch.no_grad():
                    model.dist_pred_net.network[0].bias.add_(1.)
                with self.assertRaisesRegex(ValueError, "common initialization differs"):
                    audit_scratch_initialization(model, config)
            del model

    def test_calibration_rejects_weights_source_and_r1_semantics_changes(self):
        model = torch.nn.Linear(2, 2)
        model._model_config = {"r1": {"enabled": True, "mode": "bottleneck"}}
        model._weights_origin = "ema"
        model._weights_sha256 = weights_sha256(model)
        config = {"r3": {"label_definition": {"max_path_length_m": 20.}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            artifact = {"schema_version": 2, "status": "calibrated", "temperature": 1., "threshold": .7,
                        **calibration_binding(model, config)}
            path.write_text(json.dumps(artifact))
            load_calibration(path, model, config)
            for field, changed, message in (("_weights_sha256", "different", "weights_sha256"),
                                             ("_weights_origin", "raw", "weight_source"),
                                             ("_model_config", {"r1": {"enabled": True, "mode": "auxiliary"}}, "model_config_sha256")):
                original = getattr(model, field)
                setattr(model, field, changed)
                with self.assertRaisesRegex(ValueError, message):
                    load_calibration(path, model, config)
                setattr(model, field, original)
            artifact["schema_version"] = 1
            path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, "recalibrate"):
                load_calibration(path, model, config)


class RunComparisonTest(unittest.TestCase):
    def test_mismatched_and_partial_budgets_do_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            folders = [Path(directory) / name for name in ("baseline", "r1")]
            provenance = {"training_spec": {"epochs": 1, "normalization": "window_sums"},
                          "model_config": {"context_size": 3}, "data_fingerprint": "data",
                          "initial_common_weights_sha256": "weights", "common_initialization_verified": True}
            totals = dict.fromkeys(("action_samples", "action_supervised_samples", "relation_samples", "updates", "amp_skips",
                                   "updated_action_samples", "updated_action_supervised_samples", "updated_relation_samples",
                                   "updated_visitation_valid_samples", "wall_time_s"), 1)
            for folder in folders:
                folder.mkdir()
                (folder / "provenance.json").write_text(json.dumps(provenance))
                (folder / "metrics.jsonl").write_text(json.dumps({"epoch": 0, "train": totals}) + "\n")
            self.assertTrue(compare_runs(folders)["successful_update_budgets_match"])
            totals["updates"] = 0
            (folders[1] / "metrics.jsonl").write_text(json.dumps({"epoch": 0, "train": totals}) + "\n")
            self.assertFalse(compare_runs(folders)["successful_update_budgets_match"])
            (folders[1] / "metrics.jsonl").write_text("")
            self.assertIsNone(compare_runs(folders)["consumed_action_budgets_match"])
            changed = copy.deepcopy(provenance)
            changed["initial_common_weights_sha256"] = "wrong"
            (folders[1] / "provenance.json").write_text(json.dumps(changed))
            self.assertFalse(compare_runs(folders)["configuration_and_initialization_match"])


if __name__ == "__main__":
    unittest.main()
