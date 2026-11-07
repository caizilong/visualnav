"""Semantic checks for Ubuntu execution; these do not require a simulator."""

import copy
from pathlib import Path
import tempfile
import unittest

from vint_train.data.simulation import validate_relation
from vint_train.inference.goal_relation_selector import (
    GoalRelationSelector, choose_subgoal, legal_candidate_nodes,
    localization_window, prepare_route_decision,
)
from vint_train.training.relation_metrics import binary_metrics, choose_threshold


def candidates(probabilities, distances=None):
    distances = distances if distances is not None else [1.] * len(probabilities)
    return [dict(node_index=i, probability=p, distance_m=d)
            for i, (p, d) in enumerate(zip(probabilities, distances))]


class SelectionTests(unittest.TestCase):
    def test_base_priority_and_paired_distance_counterfactual(self):
        rows = candidates([.9, .9, .9], [3., 2., 1.])
        choice = choose_subgoal(rows, 0, -1, [0, 1, 2])
        self.assertEqual(choice["selected_subgoal"], 0)
        self.assertFalse(choice["distance_changed_selection"])
        rows[0]["probability"] = .1
        choice = choose_subgoal(rows, 0, -1, [0, 1, 2])
        self.assertEqual(choice["selected_subgoal"], 2)
        self.assertEqual(choice["r_only_selected_subgoal"], 1)
        self.assertTrue(choice["distance_changed_selection"])
        rows[1]["distance_m"] = rows[2]["distance_m"]
        self.assertEqual(choose_subgoal(rows, 0, -1, [0, 1, 2])["selected_subgoal"], 1)

    def test_required_break_frontier_and_missing_candidate(self):
        route = list(range(6))
        self.assertEqual(legal_candidate_nodes(1, 0, route, required_nodes=[2]), [1, 2])
        self.assertEqual(legal_candidate_nodes(1, 0, route, broken_after=[1]), [1])
        self.assertEqual(legal_candidate_nodes(1, 2, route), [3])
        self.assertEqual(legal_candidate_nodes(4, 0, route, required_nodes=[2]), [])
        with self.assertRaisesRegex(ValueError, "Missing scores"):
            choose_subgoal(candidates([.2, .9]), 0, -1, [0, 1, 2])

    def test_localization_edge_requires_extra_forward_scores(self):
        route = list(range(12))
        nodes = localization_window(1, 1, route)
        self.assertEqual(nodes, [0, 1, 2, 3])
        prepared = prepare_route_decision(nodes, [9., 8., 7., 6.], 3., route)
        self.assertEqual(prepared["base_subgoal"], 3)
        self.assertEqual(legal_candidate_nodes(3, -1, route), [3, 4, 5])
        self.assertEqual(prepared["completed_through"], -1)

    def test_far_selection_does_not_complete_nodes(self):
        prepared = prepare_route_decision([0, 1, 2], [1., 5., 6.], 3., [0, 1, 2, 3])
        self.assertEqual(prepared["completed_through"], 0)
        selected = choose_subgoal(candidates([.1, .1, .9, .9], [1., 1., 3., 1.]),
                                  prepared["base_subgoal"], prepared["completed_through"], [0, 1, 2, 3])
        self.assertEqual(selected["selected_subgoal"], 3)
        self.assertEqual(prepared["completed_through"], 0)

    def test_reaching_required_node_advances_default_for_baseline_too(self):
        prepared = prepare_route_decision([0, 1, 2], [8., 7., 1.], 3., list(range(6)),
                                          completed_through=1, required_nodes=[2])
        self.assertEqual(prepared["base_subgoal"], 3)
        self.assertEqual(prepared["completed_through"], 2)
        result = choose_subgoal(candidates([1.] * 6), 3, 2, list(range(6)), required_nodes=[2], max_forward_skip=0)
        self.assertEqual(result["selected_subgoal"], 3)

    def test_fresh_rejections_latch_reset_and_wall_clock(self):
        now = [0.]
        selector = GoalRelationSelector(.5, 2., clock=lambda: now[0])
        rows = candidates([.1, .1, .1])
        self.assertEqual(selector.select(1, 0, 0, rows, [0, 1, 2])["status"], "abstain")
        now[0] = .1
        selector.select(1, 0, 0, rows, [0, 1, 2])
        self.assertEqual(selector.consecutive_rejections, 1)
        selector.select(2, 0, 0, rows, [0, 1, 2])
        self.assertEqual(selector.select(3, 0, 0, rows, [0, 1, 2])["status"], "failed")
        self.assertEqual(selector.select(4, 0, 0, candidates([.9] * 3), [0, 1, 2])["status"], "failed")
        selector.reset_episode()
        self.assertIsNone(selector.last_frame_time)
        self.assertEqual(selector.metrics()["failed"], 0)
        selector.select(1, 0, 0, candidates([.9] * 3), [0, 1, 2])
        now[0] += 2.01
        self.assertEqual(selector.tick()["reason"], "rgb_timeout")
        self.assertEqual(selector.metrics()["timeouts"], 1)

    def test_rejection_deadline_despite_new_frames(self):
        now = [0.]
        selector = GoalRelationSelector(.5, 2., max_rejections=99, clock=lambda: now[0])
        rows = candidates([.1] * 3)
        selector.select(1, 0, 0, rows, [0, 1, 2])
        now[0] = 1.
        selector.select(2, 0, 0, rows, [0, 1, 2])
        now[0] = 2.
        self.assertEqual(selector.tick()["reason"], "rejection_timeout")

    def test_above_h_is_not_rejection_nonfinite_is(self):
        selector = GoalRelationSelector(.5, 2.)
        rows = candidates([.9] * 3, [10., 1., 1.])
        self.assertEqual(selector.select(1, 0, 0, rows, [0, 1, 2])["status"], "selected")
        self.assertEqual(selector.metrics()["distance_above_H"], 1)
        rows[0]["distance_m"] = float("nan")
        self.assertEqual(choose_subgoal(rows, 0, -1, [0, 1, 2])["reason"], "invalid_prediction")


class CalibrationRuleTests(unittest.TestCase):
    def test_ties_minimum_acceptance_and_fdr(self):
        p = [.9] * 100 + [.8] * 5 + [.1] * 100
        y = [1] * 100 + [0] * 105
        threshold, metrics = choose_threshold(p, y)
        self.assertEqual(threshold, .9)
        self.assertEqual(metrics["accepted"], 100)
        self.assertEqual(metrics["fdr"], 0.)
        with self.assertRaises(ValueError):
            choose_threshold([.9] * 200, [1] * 100 + [0] * 100)

    def test_tie_aware_ap_and_undefined_metrics(self):
        self.assertAlmostEqual(binary_metrics([.5, .5], [1, 0])["ap"], .5)
        empty = binary_metrics([], [])
        self.assertIsNone(empty["fdr"])
        self.assertIsNone(empty["ap"])
        with self.assertRaises(ValueError):
            choose_threshold([.9] * 99 + [.1] * 20, [1] * 99 + [0] * 20)


class LabelContractTests(unittest.TestCase):
    def test_shortest_approximate_negative_and_unknown(self):
        labels = {"max_path_length_m": 2., "max_projection_error_m": .1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rgb.jpg").touch()  # The validator checks existence, not decoding.
            row = dict(pair_id="p", scene_id="s", episode_id="e", geometry_map_id="m",
                       start_pose_ref="start", goal_pose_ref="goal", query_status="found",
                       context_images=["rgb.jpg"], goal_image="rgb.jpg", reachable=1,
                       distance_valid=True, is_shortest=True, path_length_m=1.,
                       start_projection_error_m=0., goal_projection_error_m=0.)
            validate_relation(row, labels, 0, root)
            approximate = dict(row, is_shortest=False, distance_valid=False)
            validate_relation(approximate, labels, 0, root)
            too_long = dict(row, path_length_m=3., reachable=0, distance_valid=False)
            validate_relation(too_long, labels, 0, root)
            with self.assertRaises(ValueError):
                validate_relation(dict(too_long, is_shortest=False), labels, 0, root)
            unknown = dict(too_long, is_shortest=False, reachable=None)
            validate_relation(unknown, labels, 0, root)
            with self.assertRaises(ValueError):
                validate_relation(dict(row, query_status="timeout", reachable=0, distance_valid=False), labels, 0, root)
            with self.assertRaises(ValueError):
                validate_relation(dict(row, goal_projection_error_m=.2), labels, 0, root)
            missing = copy.deepcopy(row)
            del missing["reachable"]
            with self.assertRaises(ValueError):
                validate_relation(missing, labels, 0, root)


if __name__ == "__main__":
    unittest.main()
