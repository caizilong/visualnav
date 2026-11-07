"""Bounded ordered-route selection. No torch, ROS, pose, or map queries."""

import math
import time


def localization_window(previous_node, radius, route_nodes, completed_through=-1, required_nodes=()):
    """Same ordered window for the ROS adapter and offline grouped decisions."""
    positions = {n: i for i, n in enumerate(route_nodes)}
    if previous_node not in positions or type(radius) is not int or radius < 0:
        raise ValueError("Invalid localization center/radius")
    if any(n not in positions for n in required_nodes):
        raise ValueError("Required nodes must belong to the route")
    frontier = positions.get(completed_through, -1)
    pending = [positions[n] for n in required_nodes if positions[n] > frontier]
    center = min(positions[previous_node], min(pending)) if pending else positions[previous_node]
    start, end = max(center - radius, 0), min(center + radius + 1, len(route_nodes) - 1)
    if pending:
        end = min(end, min(pending))
    return route_nodes[start:end + 1]


def localize_candidates(ordered_nodes, temporal_distances, close_threshold,
                        route_nodes, completed_through=-1, required_nodes=()):
    if not ordered_nodes or len(ordered_nodes) != len(temporal_distances) or not all(math.isfinite(x) for x in temporal_distances):
        raise ValueError("Localization requires finite temporal distances for all candidates")
    i = min(range(len(ordered_nodes)), key=lambda j: temporal_distances[j])
    localized = ordered_nodes[i]
    close = temporal_distances[i] < close_threshold
    positions = {n: j for j, n in enumerate(route_nodes)}
    base = route_nodes[min(positions[localized] + int(close), len(route_nodes) - 1)]
    frontier = positions.get(completed_through, -1)
    pending = [n for n in required_nodes if frontier < positions[n] < positions[base]]
    if pending:
        base = min(pending, key=positions.get)
    return localized, base, close


def prepare_route_decision(ordered_nodes, temporal_distances, close_threshold, route_nodes,
                           completed_through=-1, required_nodes=(), broken_after=()):
    localized, base, close = localize_candidates(ordered_nodes, temporal_distances, close_threshold,
                                                route_nodes, completed_through, required_nodes)
    positions = {n: i for i, n in enumerate(route_nodes)}
    frontier, at = positions.get(completed_through, -1), positions[localized]
    if any(n not in positions for n in broken_after):
        raise ValueError("Route breaks must belong to the route")
    blocked = (any(frontier < positions[n] < at for n in required_nodes)
               or any(frontier <= positions[n] < at for n in broken_after))
    # Only localization confirms progress. Selecting a farther goal cannot do so.
    # Leave the final node eligible until an independent arrival decision.
    if close and at < len(route_nodes) - 1 and not blocked and at > frontier:
        completed_through = localized
        # A newly reached mandatory node no longer caps the default at itself.
        # The next node may need an additional RGB score outside the window.
        _, base, _ = localize_candidates(ordered_nodes, temporal_distances, close_threshold,
                                         route_nodes, completed_through, required_nodes)
    return {"localized_node": localized, "base_subgoal": base, "close": close,
            "completed_through": completed_through, "blocked_progress": blocked}


def legal_candidate_nodes(base_subgoal, completed_through, route_nodes,
                          required_nodes=(), broken_after=(), max_forward_skip=2):
    if len(set(route_nodes)) != len(route_nodes) or base_subgoal not in route_nodes:
        raise ValueError("Route must contain unique nodes and the base subgoal")
    if max_forward_skip not in (0, 1, 2):
        raise ValueError("R3 v1 permits at most two forward skips")
    positions = {node: i for i, node in enumerate(route_nodes)}
    if any(n not in positions for n in list(required_nodes) + list(broken_after)):
        raise ValueError("Required nodes / breaks must belong to the route")
    if completed_through != -1 and completed_through not in positions:
        raise ValueError("Invalid completed_through")
    start, frontier = positions[base_subgoal], positions.get(completed_through, -1)
    legal = []
    for j in range(start, min(start + max_forward_skip + 1, len(route_nodes))):
        if j <= frontier:
            continue
        if any(frontier < positions[n] < j for n in required_nodes):
            continue
        if any(frontier <= positions[n] < j for n in broken_after):
            continue
        legal.append(route_nodes[j])
    return legal


def choose_subgoal(candidates, base_subgoal, completed_through, route_nodes,
                   required_nodes=(), broken_after=(), threshold=0.5,
                   use_distance=True, max_forward_skip=2):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    positions = {node: i for i, node in enumerate(route_nodes)}
    rows = {r["node_index"]: r for r in candidates}
    if len(rows) != len(candidates) or any(n not in positions for n in rows):
        raise ValueError("Candidate indices must be unique route nodes")
    legal = legal_candidate_nodes(base_subgoal, completed_through, route_nodes,
                                  required_nodes, broken_after, max_forward_skip)
    if set(legal) - set(rows):
        raise ValueError("Missing scores for legal base/forward candidates")
    for node in legal:
        r = rows[node]
        if (not math.isfinite(r["probability"]) or not 0 <= r["probability"] <= 1
                or not math.isfinite(r["distance_m"]) or r["distance_m"] < 0):
            return {"selected_subgoal": None, "reason": "invalid_prediction", "fallback": False,
                    "base_accepted": False, "distance_changed_selection": False}
    accepted = [n for n in legal if rows[n]["probability"] >= threshold]
    base_ok = base_subgoal in accepted
    if base_ok:
        selected, reference = base_subgoal, base_subgoal
    elif accepted:
        reference = min(accepted, key=positions.get)
        selected = min(accepted, key=lambda n: (rows[n]["distance_m"], positions[n])) if use_distance else reference
    else:
        selected = reference = None
    return {"selected_subgoal": selected, "reason": "base_accepted" if base_ok else
            ("forward_fallback" if selected is not None else "all_rejected"),
            "base_accepted": base_ok, "fallback": not base_ok,
            "distance_changed_selection": selected is not None and selected != reference,
            "r_only_selected_subgoal": reference}


class GoalRelationSelector:
    def __init__(self, threshold, max_path_length_m, max_rejections=3,
                 rejection_timeout_s=2.0, stale_timeout_s=2.0,
                 use_distance=True, max_forward_skip=2, clock=time.monotonic):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        if (type(max_rejections) is not int or max_rejections < 1
                or not all(math.isfinite(x) and x > 0 for x in (rejection_timeout_s, stale_timeout_s, max_path_length_m))):
            raise ValueError("Timeouts, rejection count, and H must be positive")
        if max_forward_skip not in (0, 1, 2):
            raise ValueError("R3 v1 permits at most two forward skips")
        self.threshold, self.horizon = threshold, max_path_length_m
        self.max_rejections, self.rejection_timeout_s = max_rejections, rejection_timeout_s
        self.stale_timeout_s, self.use_distance = stale_timeout_s, use_distance
        self.max_forward_skip, self.clock = max_forward_skip, clock
        self.reset_episode()

    def reset_episode(self):
        self.last_frame_id = self.last_frame_time = self.first_rejection_time = None
        self.consecutive_rejections = 0
        self.failed = False
        self.counts = {k: 0 for k in ("decisions", "base_accepted", "fallback", "fallback_selected",
                                    "distance_changed", "abstain", "failed", "distance_above_H",
                                    "timeouts", "invalid_predictions")}
        self.last_result = {"status": "abstain", "reason": "waiting_for_rgb", "selected_subgoal": None}

    def observe_frame(self, frame_id, now=None):
        now = self.clock() if now is None else now
        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            return False
        self.last_frame_id, self.last_frame_time = frame_id, now
        return True

    def _fail(self, reason):
        if self.failed:
            return self.last_result
        self.counts["failed"] += 1
        self.counts["timeouts"] += int(reason in ("rgb_timeout", "rejection_timeout"))
        self.failed = True
        self.last_result = dict(self.last_result, status="failed", reason=reason, selected_subgoal=None)
        return self.last_result

    def fail(self, reason):
        """External RGB/control errors latch failure until explicit reset."""
        return self._fail(reason)

    def tick(self, now=None):
        now = self.clock() if now is None else now
        if self.failed:
            return self.last_result
        if self.last_frame_time is not None and now - self.last_frame_time >= self.stale_timeout_s:
            return self._fail("rgb_timeout")
        if self.first_rejection_time is not None and now - self.first_rejection_time >= self.rejection_timeout_s:
            return self._fail("rejection_timeout")
        return self.last_result

    def select(self, frame_id, localized_node, base_subgoal, candidates, route_nodes,
               completed_through=-1, required_nodes=(), broken_after=(), now=None):
        now = self.clock() if now is None else now
        # A late frame cannot revive an episode whose deadline already expired.
        self.tick(now)
        if self.failed or not self.observe_frame(frame_id, now):
            return self.last_result
        result = choose_subgoal(candidates, base_subgoal, completed_through, route_nodes,
                                required_nodes, broken_after, self.threshold,
                                self.use_distance, self.max_forward_skip)
        self.counts["decisions"] += 1
        self.counts["base_accepted"] += int(result["base_accepted"])
        self.counts["fallback"] += int(result["fallback"])
        self.counts["distance_changed"] += int(result["distance_changed_selection"])
        self.counts["invalid_predictions"] += int(result["reason"] == "invalid_prediction")
        self.counts["distance_above_H"] += sum(math.isfinite(r["distance_m"]) and r["distance_m"] > self.horizon for r in candidates)
        if result["selected_subgoal"] is None:
            self.consecutive_rejections += 1
            if self.first_rejection_time is None:
                self.first_rejection_time = now
            self.counts["abstain"] += 1
            status = "abstain"
        else:
            self.counts["fallback_selected"] += int(result["fallback"])
            self.consecutive_rejections = 0
            self.first_rejection_time = None
            status = "selected"
        self.last_result = dict(result, status=status, localized_node=localized_node,
                                base_subgoal=base_subgoal, completed_through=completed_through,
                                candidate_scores=candidates, consecutive_rejections=self.consecutive_rejections)
        if self.consecutive_rejections >= self.max_rejections:
            return self._fail("rejection_limit")
        return self.last_result

    def metrics(self):
        n = self.counts["decisions"]
        return dict(self.counts,
                    base_accept_rate=self.counts["base_accepted"] / n if n else None,
                    fallback_rate=self.counts["fallback"] / n if n else None,
                    distance_changed_selection_rate=self.counts["distance_changed"] / n if n else None,
                    distance_changed_selection_rate_on_fallback=(self.counts["distance_changed"] /
                        self.counts["fallback_selected"] if self.counts["fallback_selected"] else None))
