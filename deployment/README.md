# ROS 2 deployment

The deployment nodes use **ROS 2 Humble, rclpy and Python 3.10**. Humble's system-installed Python extensions cannot load in the existing Python 3.9 `habitat-nav` environment. Keep Habitat's simulator environment separate, or rebuild it with a compatible Python/ROS installation.

## Environment

From the repository root, in bash:

```bash
conda env create -f deployment/deployment_environment.yaml
conda activate visualnav_ros2
source /opt/ros/humble/setup.bash
python -m pip install -e train/ -e diffusion_policy/ --no-deps
python -c 'import rclpy, torch, diffusers, efficientnet_pytorch, vint_train; from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D'
```

Use `setup.zsh` when running zsh. ROS packages come from the installed Humble distribution, not from pip. The deployment environment pins the model dependencies to versions already used by this repository; GPU wheel selection may be adjusted for the target machine. No training logger packages are required by the deployment entry points.

A compatible existing Python 3.10 training environment can also run these nodes after sourcing Humble and installing the two local packages. `nomad_train` was used for local verification, with the repository roots on `PYTHONPATH`. Installing editable packages makes imports independent of the current directory.

## Start the policy and controller

Configure model/config paths in `deployment/config/models.yaml`. Its existing relative paths resolve against `deployment/src`, even when launched elsewhere. For navigation with the goal relation module, copy the run's `resolved_config.yaml` and complete checkpoint into `deployment/model_weights/goal_relation/`.

```bash
# Supply calibration for the same weights and subgoal selector settings.
bash deployment/src/navigate.sh --model nomad_goal_relation --dir topomap --calibration /path/to/calibration.json
# Goal-free exploration also supports metric simulation checkpoints.
bash deployment/src/explore.sh --model nomad_goal_relation
# Legacy weights use the legacy waypoint scale.
bash deployment/src/navigate.sh --model nomad --dir topomap
```

The wrappers start both processes and choose the controller mode from the model configuration's `action_scale_m`. Ctrl-C stops both. They do not start robot-specific base/camera drivers. Stop the previous session before starting another. For separate terminals:

```bash
python deployment/src/pd_controller.py --control-mode metric
python deployment/src/navigate.py --model nomad_goal_relation --dir topomap --calibration /path/to/calibration.json
# Or: python deployment/src/explore.py --model nomad_goal_relation
```

Use `--control-mode legacy` with legacy models. The mode is fixed for the controller process; restart it to switch. Metric controllers subscribe only to the metric command topic, so an old `/waypoint` publisher cannot drive them. Heartbeat loss always stops the robot and never changes the command mode.

## Topics and time

| Topic | ROS 2 message / meaning |
|---|---|
| `/usb_cam/image_raw` | `sensor_msgs/msg/Image`, supported encodings: rgb8, bgr8, rgba8, bgra8, mono8 |
| `/topoplan/metric_waypoint` | `std_msgs/msg/Float32MultiArray`: **[prediction_time_s, x_m, y_m]** in a single message |
| `/waypoint` | Legacy controller input; metric policies also publish XY here for visualization |
| `/topoplan/waypoint_time_s` | `std_msgs/msg/Float32`, metric prediction time for inspection |
| `/topoplan/goal_relation_stop` | `std_msgs/msg/Bool`, true stops; false is a live enable heartbeat |
| `/topoplan/reached_goal` | `std_msgs/msg/Bool`, goal status; exploration starts with false |
| `/topoplan/reset_episode` | `std_msgs/msg/Empty`, navigation episode reset |
| `/topoplan/goal_relation_status` | `std_msgs/msg/String`, subgoal selector status as JSON |
| `/cmd_vel_mux/input/navi` | `geometry_msgs/msg/Twist`, configurable via `robot.yaml` |

For waypoint index 2 and `action_time_step_s=0.2`, prediction time is **0.6 s**. The metric controller uses the duration in the command itself, then applies speed limits. The separate time topic cannot be mismatched with a command. Invalid commands, a stopped/expired heartbeat, goal arrival and expired waypoints produce zero velocity. A fresh enable and fresh waypoint are required after a stop or timeout.

Control messages are reliable and volatile. Camera subscriptions use best-effort QoS so common sensor publishers can connect. Goal/status/time publishers use transient-local durability; subscribe with matching durability to retrieve their latest state. Enable heartbeats are volatile so restarting a controller cannot replay an old enable signal.

Camera frames need **positive, strictly increasing `header.stamp.sec/nanosec`**. ROS 2 has no `header.seq`. Repeated/zero stamps do not count as new observations or keep the watchdog alive. Publish `/topoplan/reset_episode` before resetting camera timestamps. Context frames use the checkpoint's observation stride and the original frame cadence described in `docs/GOAL_RELATION_HABITAT.md`.

Watchdogs and controller expiry use monotonic wall time and remain effective when `use_sim_time=true` and `/clock` pauses. ROS remaps/parameters work through `--ros-args`; when using the paired wrappers those arguments go to the policy. Configure the controller separately if it needs different remaps.

## Hardware and topomaps

Start the base driver appropriate to your robot. The optional launch file supports ROS 2 `usb_cam`, `joy` and `twist_mux` if those packages are installed:

```bash
ros2 launch deployment/src/vint_locobot.launch.py camera:=true joystick:=true mux:=true cmd_vel_topic:=/cmd_vel
python deployment/src/joy_teleop.py
bash deployment/src/record_bag.sh demo
# After recording, build a topomap (the wrapper plays the ROS 2 bag).
bash deployment/src/create_topomap.sh topomap deployment/topomaps/bags/demo
```

All optional driver nodes default to disabled. Configure camera parameters and joystick button/axis mappings under `deployment/config/`. The ROS 2 mux outputs `/cmd_vel` by default; set `cmd_vel_topic` to your base driver's input. `create_topomap.py` refuses to replace a nonempty map unless `--overwrite` is explicitly supplied. Its output is a consecutive RGB image sequence starting at `0.png`. The legacy training rosbag-processing scripts are separate from this ROS 2 deployment path.

Habitat closed-loop control still requires a simulator adapter that publishes stamped RGB and consumes the robot's velocity commands; ROS 2 migration does not supply the simulator-specific adapter.

## Verification

Local verification on 2026-11-08: **21 GOAL_RELATION training tests and 10 ROS 2 deployment tests passed** using `nomad_train` (Python 3.10) and the installed Humble runtime. The deployment tests include actual small-model checkpoint loading and inference for exploration/navigation with synthetic RGB over DDS, plus image-timeout stopping. Launch argument loading, CLI help and shell syntax also passed. Real camera/base drivers and Habitat closed-loop navigation remain target-platform acceptance work.
