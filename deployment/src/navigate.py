"""ROS 2 visual navigation for legacy and metric simulation checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

from deployment_config import DEPLOYMENT_ROOT, model_paths, nonnegative_float, robot_config
from observation_buffer import ObservationBuffer
from ros2_runtime import ROS2Runtime, application_args
from topic_names import IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC, REACHED_GOAL_TOPIC
from utils import to_numpy, transform_images, load_model
from vint_train.inference.rgb_policy import sample_conditioned_actions

TOPOMAP_IMAGES_DIR = DEPLOYMENT_ROOT / "topomaps" / "images"
ROBOT = robot_config()
MAX_V, RATE = ROBOT["max_v"], ROBOT["frame_rate"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    paths = model_paths(args.model)
    config = yaml.safe_load(paths["config_path"].read_text())
    if not 0 <= args.waypoint < config["len_traj_pred"] or args.num_samples < 1 or args.radius < 0:
        raise ValueError("Invalid waypoint index, sample count or radius")
    model = load_model(paths["ckpt_path"], config, device, use_ema=paths.get("use_ema", True))
    if "action_scale_m" in model._model_config:
        from goal_relation_navigate import run_navigation
        return run_navigation(args, model, config, TOPOMAP_IMAGES_DIR, device)
    with ROS2Runtime("navigation") as ros:
        return run_legacy(args, model, config, ros)


@torch.no_grad()
def run_legacy(args, model, config, ros):
    directory = Path(TOPOMAP_IMAGES_DIR) / args.dir
    paths = sorted((p for p in directory.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
                   key=lambda p: int(p.stem))
    if not paths or [int(p.stem) for p in paths] != list(range(len(paths))):
        raise ValueError("Topomap images must have contiguous indices starting at zero")
    topomap = [PILImage.open(path).convert("RGB") for path in paths]
    goal_node = len(topomap) - 1 if args.goal_node == -1 else args.goal_node
    if not 0 <= goal_node < len(topomap):
        raise ValueError("Invalid goal index")
    observations = ObservationBuffer(config["context_size"])
    ros.subscribe(Image, IMAGE_TOPIC, observations.callback, image=True)
    waypoint_pub = ros.publisher(Float32MultiArray, WAYPOINT_TOPIC)
    samples_pub = ros.publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC)
    goal_pub = ros.publisher(Bool, REACHED_GOAL_TOPIC, state=True)
    goal_pub.publish(Bool(data=False))
    closest_node, last_processed, reached = 0, None, False
    print("ROS 2 navigation started. Waiting for image observations...", flush=True)
    try:
        while ros.ok():
            frame_id, frames = observations.snapshot()
            if reached or frames is None or frame_id == last_processed:
                if frames is None or reached:
                    waypoint_pub.publish(Float32MultiArray(data=[0., 0.]))
                ros.sleep(.02)
                continue
            last_processed = frame_id
            start = max(closest_node - args.radius, 0)
            end = min(closest_node + args.radius + 1, goal_node)
            nomad = config["model_type"] == "nomad"
            obs = transform_images(frames, config["image_size"], center_crop=nomad).to(device)
            goals = torch.cat([transform_images(image, config["image_size"], center_crop=nomad)
                               for image in topomap[start:end + 1]]).to(device)
            if nomad:
                condition = model("vision_encoder", obs_img=obs.repeat(len(goals), 1, 1, 1),
                                  goal_img=goals, input_goal_mask=torch.zeros(len(goals), device=device, dtype=torch.long))
                distances = to_numpy(model("dist_pred_net", obsgoal_cond=condition).flatten())
                nearest = int(np.argmin(distances))
                closest_node = start + nearest
                selected = min(nearest + int(distances[nearest] < args.close_threshold), len(goals) - 1)
                sampling = dict(model._model_config, action_scale_m=1.)
                samples = sample_conditioned_actions(model, condition[selected:selected + 1], sampling, args.num_samples)[0]
                samples = to_numpy(samples)
                chosen = samples[0, args.waypoint].copy()
                samples_pub.publish(Float32MultiArray(data=np.concatenate(([0.], samples.ravel())).tolist()))
            else:
                distances, waypoints = model(obs.repeat(len(goals), 1, 1, 1), goals)
                distances, waypoints = to_numpy(distances).flatten(), to_numpy(waypoints)
                nearest = int(np.argmin(distances))
                selected = min(nearest + int(distances[nearest] < args.close_threshold), len(waypoints) - 1)
                chosen = waypoints[selected, args.waypoint].copy()
                closest_node = start + selected
            if not np.isfinite(chosen).all() or not np.isfinite(distances).all():
                raise FloatingPointError("Non-finite navigation prediction")
            reached = start + nearest == goal_node and distances[nearest] < args.close_threshold
            goal_pub.publish(Bool(data=bool(reached)))
            if config.get("normalize", True):
                chosen[:2] *= MAX_V / RATE
            if reached or not observations.fresh():
                chosen = np.zeros(2)
            waypoint_pub.publish(Float32MultiArray(data=chosen.tolist()))
            ros.sleep(1 / RATE)
    finally:
        waypoint_pub.publish(Float32MultiArray(data=[0., 0.]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "-m", default="nomad", help="Name in deployment/config/models.yaml")
    parser.add_argument("--waypoint", "-w", type=int, default=2, help="Zero-based predicted waypoint index")
    parser.add_argument("--dir", "-d", default="topomap", help="Topomap name or absolute image directory")
    parser.add_argument("--goal-node", "-g", type=int, default=-1, help="Final topomap node; -1 means the last node")
    parser.add_argument("--close-threshold", "-t", type=nonnegative_float, default=3.0,
                        help="Temporal distance threshold; must match GOAL_RELATION calibration settings")
    parser.add_argument("--radius", "-r", type=int, default=4)
    parser.add_argument("--num-samples", "-n", type=int, default=8)
    parser.add_argument("--calibration", help="GOAL_RELATION calibration JSON matching these weights")
    parser.add_argument("--allow-uncalibrated", action="store_true", help="Explicit GOAL_RELATION debugging mode")
    parser.add_argument("--route-config", help="Optional required_nodes / broken_after JSON")
    parser.add_argument("--decision-log", default=str(DEPLOYMENT_ROOT / "logs" / "goal_relation_decisions.jsonl"))
    return parser


if __name__ == "__main__":
    try:
        main(build_parser().parse_args(application_args()))
    except KeyboardInterrupt:
        pass
