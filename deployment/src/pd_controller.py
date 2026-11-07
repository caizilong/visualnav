"""ROS 2 waypoint controller with explicit legacy/metric command modes."""

import argparse
import math
import threading
import time

import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32MultiArray

from deployment_config import robot_config
from ros2_runtime import ROS2Runtime, application_args
from ros_data import ROSData
from topic_names import METRIC_WAYPOINT_TOPIC, REACHED_GOAL_TOPIC, STOP_TOPIC, WAYPOINT_TOPIC

ROBOT = robot_config()
MAX_V, MAX_W = ROBOT["max_v"], ROBOT["max_w"]
DT = 1 / ROBOT["frame_rate"]
RATE = 9
WAYPOINT_TIMEOUT = 1.0
EPS = 1e-8


def clip_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def pd_controller(waypoint, time_step_s=None):
    values = np.asarray(waypoint, dtype=float)
    dt = DT if time_step_s is None else time_step_s
    if values.shape not in ((2,), (4,)) or not np.isfinite(values).all():
        raise ValueError("waypoint must be a finite XY or XY-heading vector")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("prediction time must be finite and positive")
    dx, dy = values[:2]
    if len(values) == 4 and abs(dx) < EPS and abs(dy) < EPS:
        v, w = 0., clip_angle(np.arctan2(values[3], values[2])) / dt
    elif abs(dx) < EPS:
        v, w = 0., np.sign(dy) * np.pi / (2 * dt)
    else:
        v, w = dx / dt, np.arctan(dy / dx) / dt
    return float(np.clip(v, 0, MAX_V)), float(np.clip(w, -MAX_W, MAX_W))


class ControlState:
    """Timeouts and stop signals use monotonic time, independently of ROS time."""

    def __init__(self, mode, clock=time.monotonic):
        if mode not in ("legacy", "metric"):
            raise ValueError("control mode must be legacy or metric")
        self.mode, self.clock = mode, clock
        self.lock = threading.RLock()
        self.waypoint = ROSData(WAYPOINT_TIMEOUT, clock=clock)
        self.duration = None
        self.reached = False
        self.stopped = True
        self.heartbeat = float("-inf")

    def receive_waypoint(self, values):
        with self.lock:
            values = np.asarray(values, dtype=float)
            if self.mode == "metric":
                if values.shape != (3,) or not np.isfinite(values).all() or values[0] <= 0:
                    self.waypoint.clear()
                    return
                if self.stopped or self.clock() - self.heartbeat >= WAYPOINT_TIMEOUT:
                    self.waypoint.clear()
                    return
                self.duration, values = float(values[0]), values[1:]
            elif values.shape not in ((2,), (4,)) or not np.isfinite(values).all():
                self.waypoint.clear()
                return
            if not self.reached:
                self.waypoint.set(values.copy())

    def receive_stop(self, stopped):
        with self.lock:
            expired = self.clock() - self.heartbeat >= WAYPOINT_TIMEOUT
            self.heartbeat = self.clock()
            if stopped or self.stopped or expired:
                self.waypoint.clear()
            self.stopped = bool(stopped)

    def receive_goal(self, reached):
        with self.lock:
            if reached or self.reached != bool(reached):
                self.waypoint.clear()
            self.reached = bool(reached)

    def command(self):
        with self.lock:
            valid = not self.reached and self.waypoint.is_valid()
            if self.mode == "metric":
                valid = valid and not self.stopped and self.clock() - self.heartbeat < WAYPOINT_TIMEOUT
            if not valid:
                self.waypoint.clear()
                return 0., 0.
            return pd_controller(self.waypoint.get(), self.duration if self.mode == "metric" else None)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-mode", choices=("legacy", "metric"), default="legacy",
                        help="Must match the policy; restart to change command modes")
    parsed = parser.parse_args(application_args() if args is None else args)
    state = ControlState(parsed.control_mode)
    with ROS2Runtime("pd_controller") as ros:
        publisher = ros.publisher(Twist, ROBOT["vel_navi_topic"])

        def stop(message):
            with state.lock:
                state.receive_stop(message.data)
                if message.data:
                    publisher.publish(Twist())

        def goal(message):
            with state.lock:
                state.receive_goal(message.data)
                if message.data:
                    publisher.publish(Twist())

        topic = METRIC_WAYPOINT_TOPIC if state.mode == "metric" else WAYPOINT_TOPIC
        ros.subscribe(Float32MultiArray, topic, lambda msg: state.receive_waypoint(msg.data))
        ros.subscribe(Bool, REACHED_GOAL_TOPIC, goal, state=True)
        if state.mode == "metric":
            ros.subscribe(Bool, STOP_TOPIC, stop)  # Never replay an old enable heartbeat.
        print(f"ROS 2 controller: mode={state.mode}, input={topic}", flush=True)
        try:
            while ros.ok():
                with state.lock:
                    v, w = state.command()
                    message = Twist()
                    message.linear.x, message.angular.z = v, w
                    publisher.publish(message)
                ros.sleep(1 / RATE)
        finally:
            publisher.publish(Twist())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
