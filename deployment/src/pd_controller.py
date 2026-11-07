import numpy as np
import yaml
import time
import threading
from typing import Tuple

# ROS
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Float32MultiArray, Bool

from topic_names import (WAYPOINT_TOPIC, 
			 			REACHED_GOAL_TOPIC)
from ros_data import ROSData
from utils import clip_angle

# CONSTS
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1/robot_config["frame_rate"]
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
FLIP_ANG_VEL = np.pi/4

# GLOBALS
vel_msg = Twist()
waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
reached_goal = False
reverse_mode = False
current_yaw = None
r3_waypoint = ROSData(WAYPOINT_TIMEOUT, name="r3_waypoint", clock=time.monotonic)
r3_active = False
r3_stop = True
r3_heartbeat = float("-inf")
control_lock = threading.RLock()
velocity_publisher = None
r3_waypoint_time_s = None

def clip_angle(theta) -> float:
	"""Clip angle to [-pi, pi]"""
	theta %= 2 * np.pi
	if -np.pi < theta < np.pi:
		return theta
	return theta - 2 * np.pi
      

def pd_controller(waypoint: np.ndarray, time_step_s=None) -> Tuple[float]:
	"""PD controller for the robot"""
	assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
	dt = DT if time_step_s is None else time_step_s
	if len(waypoint) == 2:
		dx, dy = waypoint
	else:
		dx, dy, hx, hy = waypoint
	# this controller only uses the predicted heading if dx and dy near zero
	if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
		v = 0
		w = clip_angle(np.arctan2(hy, hx))/dt
	elif np.abs(dx) < EPS:
		v =  0
		w = np.sign(dy) * np.pi/(2*dt)
	else:
		v = dx / dt
		w = np.arctan(dy/dx) / dt
	v = np.clip(v, 0, MAX_V)
	w = np.clip(w, -MAX_W, MAX_W)
	return v, w


def callback_drive(waypoint_msg: Float32MultiArray):
	"""Callback function for the waypoint subscriber"""
	global vel_msg
	with control_lock:
		if r3_active:
			values = np.asarray(waypoint_msg.data)
			if not r3_stop and len(values) in (2, 4) and np.isfinite(values).all():
				r3_waypoint.set(values)
		else:
			waypoint.set(waypoint_msg.data)
	
	
def callback_reached_goal(reached_goal_msg: Bool):
	"""Callback function for the reached goal subscriber"""
	global reached_goal
	with control_lock:
		reached_goal = reached_goal_msg.data
		if reached_goal and velocity_publisher is not None:
			velocity_publisher.publish(Twist())


def callback_r3_stop(message: Bool):
	global r3_active, r3_stop, r3_heartbeat
	with control_lock:
		r3_active = True
		r3_heartbeat = time.monotonic()
		if message.data or r3_stop:
			r3_waypoint.last_time_received = float("-inf")
		r3_stop = message.data
		if r3_stop and velocity_publisher is not None:
			# Callback threads remain responsive even if ROS simulation time pauses.
			velocity_publisher.publish(Twist())


def callback_waypoint_time(message: Float32):
	global r3_waypoint_time_s
	with control_lock:
		r3_waypoint_time_s = message.data if np.isfinite(message.data) and message.data > 0 else None


def main():
	global vel_msg, reverse_mode, velocity_publisher
	rospy.init_node("PD_CONTROLLER", anonymous=False)
	waypoint_sub = rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, callback_drive, queue_size=1)
	reached_goal_sub = rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, callback_reached_goal, queue_size=1)
	rospy.Subscriber("/topoplan/r3_stop", Bool, callback_r3_stop, queue_size=1)
	rospy.Subscriber("/topoplan/waypoint_time_s", Float32, callback_waypoint_time, queue_size=1)
	vel_out = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
	velocity_publisher = vel_out
	rate = rospy.Rate(RATE)
	print("Registered with master node. Waiting for waypoints...")
	while not rospy.is_shutdown():
		if r3_active:
			with control_lock:
				command = Twist()
				if (not reached_goal and not r3_stop and time.monotonic() - r3_heartbeat < WAYPOINT_TIMEOUT
						and r3_waypoint.is_valid() and r3_waypoint_time_s is not None):
					v, w = pd_controller(r3_waypoint.get(), r3_waypoint_time_s)
					command.linear.x = -v if reverse_mode else v
					command.angular.z = w
				vel_out.publish(command)
			time.sleep(1 / RATE)
			continue
		vel_msg = Twist()
		if reached_goal:
			vel_out.publish(vel_msg)
			print("Reached goal! Stopping...")
			return
		elif waypoint.is_valid(verbose=True):
			v, w = pd_controller(waypoint.get())
			if reverse_mode:
				v *= -1
			vel_msg.linear.x = v
			vel_msg.angular.z = w
			print(f"publishing new vel: {v}, {w}")
		vel_out.publish(vel_msg)
		rate.sleep()
	

if __name__ == '__main__':
	main()
