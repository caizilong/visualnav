"""ROS 2 goal-free NoMaD exploration with legacy or metric waypoints."""

import argparse
import threading
import time

import numpy as np
import torch
import yaml
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray

from deployment_config import model_paths, robot_config
from observation_buffer import ObservationBuffer
from ros2_runtime import ROS2Runtime, application_args
from topic_names import (IMAGE_TOPIC, METRIC_WAYPOINT_TOPIC, REACHED_GOAL_TOPIC,
                         SAMPLED_ACTIONS_TOPIC, STOP_TOPIC, WAYPOINT_TIME_TOPIC, WAYPOINT_TOPIC)
from utils import load_model, transform_images
from vint_train.inference.rgb_policy import sample_conditioned_actions


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "-m", default="nomad")
    parser.add_argument("--waypoint", "-w", default=2, type=int)
    parser.add_argument("--num-samples", "-n", default=8, type=int)
    return parser


def main(args):
    paths = model_paths(args.model)
    config = yaml.safe_load(paths["config_path"].read_text())
    if config["model_type"] != "nomad":
        raise ValueError("Exploration requires a NoMaD model")
    if not 0 <= args.waypoint < config["len_traj_pred"] or args.num_samples < 1:
        raise ValueError("Invalid waypoint index or sample count")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(paths["ckpt_path"], config, device, use_ema=paths.get("use_ema", True))
    canonical = model._model_config
    metric = "action_scale_m" in canonical
    duration = (args.waypoint + 1) * canonical["action_time_step_s"] if metric else None
    sampling = dict(canonical, action_scale_m=canonical.get("action_scale_m", 1.))
    robot = robot_config()
    timeout = config.get("goal_relation", {}).get("runtime", {}).get("stale_timeout_s", 2.)
    observations = ObservationBuffer(config["context_size"], canonical.get("observation_stride", 1), timeout)
    control_lock = threading.RLock()
    enabled = False
    stop_event = threading.Event()

    with ROS2Runtime("exploration") as ros:
        waypoint_pub = ros.publisher(Float32MultiArray, WAYPOINT_TOPIC)
        samples_pub = ros.publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC)
        goal_pub = ros.publisher(Bool, REACHED_GOAL_TOPIC, state=True)
        goal_pub.publish(Bool(data=False))
        ros.subscribe(Image, IMAGE_TOPIC, observations.callback, image=True)
        if metric:
            metric_pub = ros.publisher(Float32MultiArray, METRIC_WAYPOINT_TOPIC)
            duration_pub = ros.publisher(Float32, WAYPOINT_TIME_TOPIC, state=True)
            stop_pub = ros.publisher(Bool, STOP_TOPIC)
            duration_pub.publish(Float32(data=float(duration)))
            stop_pub.publish(Bool(data=True))

        def watchdog():
            while not stop_event.wait(.05) and ros.ok():
                with control_lock:
                    stop_pub.publish(Bool(data=not (enabled and observations.fresh())))

        thread = threading.Thread(target=watchdog, daemon=True) if metric else None
        if thread:
            thread.start()
        last_processed = None
        try:
            while ros.ok():
                frame_id, frames = observations.snapshot()
                if frames is None or frame_id == last_processed:
                    ros.sleep(.02)
                    continue
                last_processed = frame_id
                started = time.monotonic()
                with torch.no_grad():
                    obs = transform_images(frames, config["image_size"], center_crop=True).to(device)
                    width, height = canonical["image_size"]
                    fake_goal = torch.zeros((1, 3, height, width), device=device)
                    condition = model("vision_encoder", obs_img=obs, goal_img=fake_goal,
                                      input_goal_mask=torch.ones(1, device=device, dtype=torch.long))
                    samples = sample_conditioned_actions(model, condition, sampling, args.num_samples)[0].cpu().numpy()
                if not np.isfinite(samples).all():
                    raise FloatingPointError("Non-finite exploration actions")
                if not observations.fresh() or time.monotonic() - started >= timeout:
                    with control_lock:
                        enabled = False
                        if metric:
                            stop_pub.publish(Bool(data=True))
                    continue
                chosen = samples[0, args.waypoint].copy()
                if config.get("normalize", True) and not metric:
                    chosen *= robot["max_v"] / robot["frame_rate"]
                with control_lock:
                    if metric:
                        enabled = True
                        stop_pub.publish(Bool(data=False))
                        duration_pub.publish(Float32(data=float(duration)))
                        metric_pub.publish(Float32MultiArray(data=[float(duration), *chosen.tolist()]))
                    waypoint_pub.publish(Float32MultiArray(data=chosen.tolist()))
                    samples_pub.publish(Float32MultiArray(data=np.concatenate(([0.], samples.ravel())).tolist()))
                ros.sleep(1 / robot["frame_rate"])
        finally:
            stop_event.set()
            if thread:
                thread.join(timeout=1.)
            with control_lock:
                if metric:
                    stop_pub.publish(Bool(data=True))
                waypoint_pub.publish(Float32MultiArray(data=[0., 0.]))


if __name__ == "__main__":
    try:
        main(build_parser().parse_args(application_args()))
    except KeyboardInterrupt:
        pass
