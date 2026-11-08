"""R1 regression checks for Ubuntu. Added without executing in the Mac session."""

import copy
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vint_train.data.simulation import file_sha256
from vint_train.data.trajectory_regions import fit_region_statistics, load_region_statistics
from vint_train.models.nomad.spatial_bottleneck import SpatialBottleneck
from vint_train.training.trajectory_visitation import visitation_targets, visitation_loss_sum, VisitationMetrics


def definition():
    return {"schema_version": 1, "target": "future_waypoint_visit_distribution",
            "distance_unit": "meter", "coordinate_frame": "nomad_xy_ccw",
            "angular_convention": "uniform_centered_forward_ccw",
            "radial_boundary_rule": "left_closed_right_open", "waypoint_weighting": "uniform",
            "num_sectors": 6, "num_radial_bins": 4, "stationary_epsilon_m": 1.e-4,
            "radial_boundaries_m": [1., 2., 3.]}


class VisitationTargetsTest(unittest.TestCase):
    def test_metric_scale_boundaries_and_invalid_observations(self):
        points = torch.tensor([[[.25, 0], [.5, 0], [1., 0], [2., 0], [-.25, 0], [0., 0]]])
        actions = torch.cat((points, torch.ones_like(points) * 99, torch.zeros_like(points)))
        batch = (None, None, actions, None, None, None, torch.tensor([1., 0., 1.]))
        q, valid, count = visitation_targets(batch, {"action_scale_m": 2., "r1": {"discretization": definition()}})
        self.assertEqual(valid.tolist(), [True, False, False])
        self.assertEqual(count.tolist(), [5, 0, 0])
        expected = torch.zeros(6, 4)
        expected[0] = .2  # Forward: .5, 1, 2 and 4 meters.
        expected[3, 0] = .2  # Rear: .5 meters.
        torch.testing.assert_close(q[0], expected)
        self.assertEqual(float(q[1:].sum()), 0.)

    def test_invalid_logits_are_excluded_before_reduction(self):
        logits = torch.zeros(3, 6, 4, requires_grad=True)
        with torch.no_grad():
            logits[1] = float("nan")
        q = torch.zeros_like(logits)
        q[0, 0, 0], q[2, 5, 3] = 1., 1.
        valid = torch.tensor([True, False, True])
        loss = visitation_loss_sum(logits, q, valid)
        self.assertAlmostEqual(float(loss), 2 * math.log(24), places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.)

    def test_uneven_microbatches_match_one_window_gradient(self):
        generator = torch.Generator().manual_seed(5)
        logits = torch.randn(5, 6, 4, generator=generator, requires_grad=True)
        split_logits = logits.detach().clone().requires_grad_()
        q = torch.zeros_like(logits)
        q[:, 0, 1], q[:, 1, 2] = .25, .75
        valid = torch.tensor([True, False, True, True, False])
        visitation_loss_sum(logits, q, valid).div(valid.sum()).backward()
        for start, stop in ((0, 2), (2, 5)):
            visitation_loss_sum(split_logits[start:stop], q[start:stop], valid[start:stop]).div(valid.sum()).backward()
        torch.testing.assert_close(logits.grad, split_logits.grad)
        empty = visitation_loss_sum(logits, q, torch.zeros_like(valid))
        self.assertEqual(float(empty), 0.)
        self.assertFalse(empty.requires_grad)

    def test_untrained_metrics_do_not_report_zero_error(self):
        metrics = VisitationMetrics(head_trained=False)
        metrics.update(None, torch.zeros(2, 6, 4), torch.tensor([True, False]), torch.tensor([8, 0]))
        report = metrics.report()
        self.assertEqual(report["valid_samples"], 1)
        self.assertIsNone(report["cross_entropy"])
        self.assertIsNone(report["kl_target_to_prediction"])


class SpatialBottleneckTest(unittest.TestCase):
    def test_identity_initialization_and_supervised_gradient(self):
        torch.manual_seed(7)
        module = SpatialBottleneck(feature_dim=12, condition_dim=16, token_dim=8, num_attention_heads=2)
        features, condition = torch.randn(2, 12, 3, 4), torch.randn(2, 16)
        tokens = module.encode(features)
        self.assertEqual(tuple(tokens.shape), (2, 6, 8))
        torch.testing.assert_close(module.fuse(condition, tokens), condition, rtol=0, atol=0)
        logits = module.predict_visitation(tokens, condition)
        self.assertEqual(tuple(logits.shape), (2, 6, 4))
        targets = torch.zeros_like(logits)
        targets[0, 0, 0], targets[1, 4, 3] = 1., 1.
        visitation_loss_sum(logits, targets, torch.ones(2, dtype=torch.bool)).backward()
        for parameter in (module.queries, module.spatial_projection.weight, module.visitation_head[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)


class RegionStatisticsTest(unittest.TestCase):
    def test_train_only_fit_and_stale_sampling_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "train/episode/traj_data.pkl"
            trajectory.parent.mkdir(parents=True)
            with trajectory.open("wb") as stream:
                pickle.dump({"position": np.column_stack((np.arange(25.), np.zeros(25))),
                             "yaw": np.zeros(25)}, stream)
            bundle = {"root": root, "trajectories": {"train": [{"name": "train/episode"}]},
                      "manifest": {"waypoint_spacing": 1, "action_scale_m": .5,
                                   "action_time_step_s": .2, "data_revision": "unit-fixture"},
                      "file_hashes": {"trajectory/train/episode": file_sha256(trajectory)}}
            config = {"context_size": 1, "len_traj_pred": 4, "distance": {"max_dist_cat": 8},
                      "action": {"min_dist_cat": 1, "max_dist_cat": 8}, "r1": {"max_fit_points": 1000}}
            statistics = fit_region_statistics(bundle, config)
            # An unreadable validation path must not affect the fitted boundaries.
            bundle["trajectories"]["validation"] = [{"name": "does-not-exist"}]
            self.assertEqual(statistics, fit_region_statistics(bundle, config))
            output = root / "r1_statistics.json"
            output.write_text(json.dumps(statistics))
            loaded, _, _ = load_region_statistics(output, bundle, config)
            self.assertEqual(loaded, statistics["definition"])
            changed = copy.deepcopy(config)
            changed["len_traj_pred"] = 5
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_region_statistics(output, bundle, changed)


if __name__ == "__main__":
    unittest.main()
