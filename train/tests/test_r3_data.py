"""Train-only normalization and cross-stream scene leakage checks (deferred)."""

import json
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vint_train.data.prepare_sim_data import compute_statistics
from vint_train.data.simulation import load_manifest, load_action_statistics


class SimulationDataTests(unittest.TestCase):
    def write_export(self, root):
        label = dict(label_schema_version=1, distance_unit="meter", max_path_length_m=2.,
                     robot_footprint_id="footprint_v1", planner_config_id="planner_v1",
                     motion_constraint_id="static_footprint_holonomic", unknown_space_policy="unknown",
                     query_region="full_scene_navmesh", max_projection_error_m=.1)
        manifest = dict(schema_version=1, platform="habitat", data_revision="immutable_fixture",
                        data_root="rgb", coordinate_frame="nomad_xy_ccw", action_scale_m=.5,
                        action_time_step_s=.2, waypoint_spacing=1, label_definition=label, splits={})
        for split, positions in (("train", [[0., 0.], [1., .5], [3., 1.5], [2., .5]]),
                                 ("validation", [[0., 0.], [100., 100.], [-10., -30.], [0., 200.]])):
            trajectory = root / "rgb" / split
            trajectory.mkdir(parents=True)
            with (trajectory / "traj_data.pkl").open("wb") as stream:
                pickle.dump({"position": np.array(positions), "yaw": np.zeros(4)}, stream)
            for t in range(4):
                (trajectory / f"{t}.jpg").touch()
            manifest["splits"][split] = {"trajectories": [dict(name=split, scene_id=split, episode_id="e", geometry_map_id=split + "_map")]}
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path, manifest

    def test_statistics_ignore_validation_and_bind_training_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = self.write_export(root)
            bundle = load_manifest(path, context_size=0)
            stats = compute_statistics(bundle, 0, 1)
            np.testing.assert_allclose(stats["action_stats"]["min"], [-2., -2.])
            np.testing.assert_allclose(stats["action_stats"]["max"], [4., 2.])
            self.assertEqual(stats["num_windows"], 3)
            output = root / "stats.json"
            output.write_text(json.dumps(stats))
            load_action_statistics(output, bundle, 0, 1)
            with self.assertRaises(ValueError):
                load_action_statistics(output, bundle, 1, 1)
            manifest["action_time_step_s"] = .4
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_action_statistics(output, load_manifest(path, 0), 0, 1)

    def test_scene_and_reused_rgb_cannot_cross_stream_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = self.write_export(root)
            row = dict(pair_id="leak", scene_id="train", episode_id="e", geometry_map_id="train_map",
                       context_images=["train/0.jpg"], goal_image="train/1.jpg",
                       start_pose_ref="p0", goal_pose_ref="p1", query_status="found", reachable=1,
                       path_length_m=1., is_shortest=True, distance_valid=True,
                       start_projection_error_m=0., goal_projection_error_m=0.)
            relations = root / "validation.jsonl"
            relations.write_text(json.dumps(row) + "\n")
            manifest["splits"]["validation"]["relations"] = "validation.jsonl"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(path, 0)
            # Relabeling the scene does not conceal reused action-training RGB.
            row["scene_id"] = "validation"
            relations.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_manifest(path, 0)


if __name__ == "__main__":
    unittest.main()
