"""ROS reference adapter for scratch simulation checkpoints (all geometry stays offline)."""

from collections import deque
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import rospy
import torch
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty, Float32, Float32MultiArray, String

from utils import msg_to_pil
from topic_names import IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC, REACHED_GOAL_TOPIC
from vint_train.data.data_utils import resize_and_aspect_crop
from vint_train.inference.calibration import load_calibration, selector_spec
from vint_train.inference.goal_relation_selector import (GoalRelationSelector, localization_window,
                                                         prepare_route_decision, legal_candidate_nodes)
from vint_train.inference.rgb_policy import score_rgb_pairs, sample_conditioned_actions


STOP_TOPIC = "/topoplan/r3_stop"
STATUS_TOPIC = "/topoplan/r3_status"
RESET_TOPIC = "/topoplan/reset_episode"


def run_navigation(args, model, config, topomap_root, device):
    spec = selector_spec(config)
    if args.radius != spec["radius"] or args.close_threshold != spec["close_threshold"]:
        raise ValueError("CLI radius/close-threshold must match the saved decision settings")
    enabled = config.get("r3", {}).get("enabled", False)
    if enabled and args.calibration:
        artifact = load_calibration(args.calibration, model, config)
        temperature, threshold = artifact["temperature"], artifact["threshold"]
        calibrated = True
    elif enabled and not args.allow_uncalibrated:
        raise ValueError("Supply --calibration, or explicitly use --allow-uncalibrated for debugging")
    else:
        temperature, threshold, calibrated = 1., .5, False
    horizon = config["r3"]["label_definition"]["max_path_length_m"]
    runtime = config.get("r3", {}).get("runtime", {})
    selector = GoalRelationSelector(threshold, horizon, runtime.get("max_rejections", 3),
                                    runtime.get("rejection_timeout_s", 2.), runtime.get("stale_timeout_s", 2.),
                                    spec["use_distance"] if enabled else False, spec["max_forward_skip"] if enabled else 0)
    directory = Path(topomap_root) / args.dir
    paths = sorted([p for p in directory.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")], key=lambda p: int(p.stem))
    if [int(p.stem) for p in paths] != list(range(len(paths))) or not paths:
        raise ValueError("Topomap RGB files must have contiguous node indices starting at zero")
    goal_node = len(paths) - 1 if args.goal_node == -1 else args.goal_node
    if not 0 <= goal_node < len(paths) or not 0 <= args.waypoint < config["len_traj_pred"]:
        raise ValueError("Invalid final goal or waypoint index")
    route = {"required_nodes": [], "broken_after": []}
    route_path = Path(args.route_config) if args.route_config else directory / "route.json"
    if route_path.exists():
        route.update(json.loads(route_path.read_text()))
    route_nodes = list(range(goal_node + 1))
    for field in ("required_nodes", "broken_after"):
        if any(type(n) is not int or n not in route_nodes for n in route[field]):
            raise ValueError(f"Invalid {field} in route metadata")
    goal_tensors = torch.stack([resize_and_aspect_crop(PILImage.open(p).convert("RGB"), config["image_size"]) for p in paths])
    queue = deque(maxlen=config["context_size"] + 1)
    lock = threading.RLock()
    state = {"frame_id": 0, "sampled_frame_id": 0, "last_raw_time": None,
             "closest": 0, "completed": -1, "reached": False, "episode": 0, "image_identity": None}
    stride = config.get("observation_stride", 1)
    stop_event = threading.Event()
    rospy.init_node("R3_NAVIGATION", anonymous=False)
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
    samples_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher(REACHED_GOAL_TOPIC, Bool, queue_size=1, latch=True)
    stop_pub = rospy.Publisher(STOP_TOPIC, Bool, queue_size=1, latch=True)
    duration_pub = rospy.Publisher("/topoplan/waypoint_time_s", Float32, queue_size=1, latch=True)
    status_pub = rospy.Publisher(STATUS_TOPIC, String, queue_size=1, latch=True)
    log_path = Path(args.decision_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def write_record(record):
        def sanitize(value):
            if isinstance(value, dict):
                return {k: sanitize(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [sanitize(v) for v in value]
            return None if isinstance(value, float) and not math.isfinite(value) else value
        # All callers hold the state lock, including the watchdog/reset callbacks.
        with log_path.open("a") as stream:
            stream.write(json.dumps(sanitize(dict(record, monotonic_s=time.monotonic())), allow_nan=False) + "\n")

    def publish_stop():
        stop_pub.publish(Bool(data=True))
        waypoint_pub.publish(Float32MultiArray(data=[0., 0.]))

    def observation_callback(message):
        with lock:
            stamp = message.header.stamp.to_nsec()
            identity = ("stamp", stamp) if stamp > 0 else ("seq", message.header.seq)
            last = state["image_identity"]
            if last is not None and identity[0] == last[0] and identity[1] <= last[1]:
                return  # Retransmitted/stale frames cannot count as new rejections.
            image = msg_to_pil(message).convert("RGB")
            state["image_identity"] = identity
            state["frame_id"] += 1
            state["last_raw_time"] = time.monotonic()
            if (state["frame_id"] - 1) % stride == 0:
                queue.append(image)
                state["sampled_frame_id"] = state["frame_id"]

    def reset_callback(_):
        with lock:
            previous_metrics = selector.metrics()
            selector.reset_episode()
            queue.clear()
            state.update(closest=0, completed=-1, reached=False, last_raw_time=None,
                         episode=state["episode"] + 1, image_identity=None)
            goal_pub.publish(Bool(data=False))
            publish_stop()
            write_record({"event": "reset", "episode": state["episode"],
                          "previous_episode": state["episode"] - 1, "previous_metrics": previous_metrics})

    def watchdog():
        last_status = None
        while not stop_event.wait(.05) and not rospy.is_shutdown():
            with lock:
                if state["reached"]:
                    publish_stop()
                    continue
                now = time.monotonic()
                if state["last_raw_time"] is not None and now - state["last_raw_time"] >= selector.stale_timeout_s:
                    selector.fail("rgb_timeout")
                status = selector.tick(now)
                if status["status"] != "selected":
                    publish_stop()
                else:
                    # Heartbeat permits the controller to stop if this node dies.
                    stop_pub.publish(Bool(data=False))
                status_pub.publish(String(data=json.dumps({"episode": state["episode"], "status": status["status"],
                                                          "reason": status["reason"], "metrics": selector.metrics()})))
                current = (state["episode"], status["status"], status["reason"])
                if current != last_status:
                    write_record({"event": "state", "episode": state["episode"], "status": status["status"],
                                  "reason": status["reason"], "metrics": selector.metrics()})
                    last_status = current

    rospy.Subscriber(IMAGE_TOPIC, Image, observation_callback, queue_size=1)
    rospy.Subscriber(RESET_TOPIC, Empty, reset_callback, queue_size=1)
    goal_pub.publish(Bool(data=False))
    publish_stop()
    duration_pub.publish(Float32(data=(args.waypoint + 1) * config["action_time_step_s"]))
    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    last_processed = None
    print(f"R3={enabled}, calibrated={calibrated}, action_scale_m={config['action_scale_m']}; RGB only")
    try:
        while not rospy.is_shutdown():
            with lock:
                key = (state["episode"], state["sampled_frame_id"])
                ready = len(queue) == config["context_size"] + 1 and key != last_processed and not state["reached"] and not selector.failed
                snapshot = list(queue) if ready else None
                episode, frame_id = key
                previous, completed = state["closest"], state["completed"]
            if snapshot is None:
                stop_event.wait(.02)
                continue
            last_processed = key
            nodes = localization_window(previous, args.radius, route_nodes, completed, route["required_nodes"])
            obs = torch.cat([resize_and_aspect_crop(im, config["image_size"]) for im in snapshot])[None].to(device)
            scored = score_rgb_pairs(model, obs.repeat(len(nodes), 1, 1, 1), goal_tensors[nodes].to(device), temperature)
            distances = scored["temporal_distance"].cpu().tolist()
            prepared = prepare_route_decision(nodes, distances, args.close_threshold, route_nodes, completed,
                                              route["required_nodes"], route["broken_after"])
            localized, base, close = prepared["localized_node"], prepared["base_subgoal"], prepared["close"]
            completed, blocked_progress = prepared["completed_through"], prepared["blocked_progress"]
            legal = legal_candidate_nodes(base, completed, route_nodes, route["required_nodes"],
                                           route["broken_after"], selector.max_forward_skip)
            extra = [n for n in legal if n not in nodes]
            if extra:
                extra_scores = score_rgb_pairs(model, obs.repeat(len(extra), 1, 1, 1), goal_tensors[extra].to(device), temperature)
                scored = {key: torch.cat((value, extra_scores[key]), dim=0) for key, value in scored.items()}
                nodes = nodes + extra
            probabilities = scored["probability"].cpu().tolist() if enabled else [1.] * len(nodes)
            metric_distances = scored["distance_m"].cpu().tolist() if enabled else [0.] * len(nodes)
            candidates = [{"node_index": n, "probability": p, "distance_m": d} for n, p, d in zip(nodes, probabilities, metric_distances)]
            with lock:
                if state["episode"] != episode:
                    continue
                choice = selector.select(frame_id, localized, base, candidates, route_nodes, completed,
                                         route["required_nodes"], route["broken_after"])
                state["closest"], state["completed"] = localized, completed
                if choice["status"] != "selected":
                    publish_stop()
                    goal_pub.publish(Bool(data=False))
                reached = (choice["status"] == "selected" and localized == goal_node and close and not blocked_progress)
                if reached:
                    state["reached"] = True
                    goal_pub.publish(Bool(data=True))
                    publish_stop()
            executed = False
            if choice["status"] == "selected" and not reached:
                index = nodes.index(choice["selected_subgoal"])
                samples = sample_conditioned_actions(model, scored["condition"][index:index+1], config, args.num_samples)[0].cpu().numpy()
                waypoint = samples[0, args.waypoint]
                if not np.isfinite(samples).all():
                    raise FloatingPointError("Non-finite diffusion output")
                with lock:
                    if state["episode"] != episode:
                        publish_stop()
                        continue
                    selector.tick()
                    if state["episode"] == episode and not selector.failed and not state["reached"]:
                        stop_pub.publish(Bool(data=False))
                        samples_pub.publish(Float32MultiArray(data=np.concatenate(([0.], samples.ravel())).tolist()))
                        waypoint_pub.publish(Float32MultiArray(data=waypoint.tolist()))
                        executed = True
                    else:
                        publish_stop()
            with lock:
                record = dict(choice, event="decision", episode=episode, frame_id=frame_id, goal_reached=reached,
                              calibrated=calibrated, weights_sha256=model._weights_sha256,
                              waypoint_published=executed, execution_status=selector.last_result["status"],
                              metrics=selector.metrics())
                write_record(record)
    except Exception as exc:
        with lock:
            status = selector.fail("runtime_error")
            publish_stop()
            goal_pub.publish(Bool(data=False))
            write_record({"event": "error", "episode": state["episode"], "status": status["status"],
                          "error": str(exc), "metrics": selector.metrics()})
        raise
    finally:
        stop_event.set()
        publish_stop()
        thread.join(timeout=1.)
