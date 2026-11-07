"""ROS 2 deadman joystick teleoperation; stop on release or input timeout."""

import threading
import time

import yaml
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from deployment_config import CONFIG_ROOT, robot_config
from ros2_runtime import ROS2Runtime
from topic_names import JOY_BUMPER_TOPIC


def main():
    robot = robot_config()
    config = yaml.safe_load((CONFIG_ROOT / "joystick.yaml").read_text())
    lock = threading.RLock()
    state = {"message": Twist(), "pressed": False, "last": float("-inf"), "bumper": False}
    with ROS2Runtime("joy_teleop") as ros:
        publisher = ros.publisher(Twist, robot["vel_teleop_topic"])
        bumper_pub = ros.publisher(Bool, JOY_BUMPER_TOPIC)

        def callback(message):
            with lock:
                deadman = config["deadman_switch"]
                linear, angular = config["lin_vel_button"], config["ang_vel_button"]
                valid = len(message.buttons) > deadman and len(message.axes) > max(linear, angular)
                pressed = valid and bool(message.buttons[deadman])
                command = Twist()
                if pressed:
                    command.linear.x = float(robot["max_v"] * message.axes[linear])
                    command.angular.z = float(robot["max_w"] * message.axes[angular])
                state.update(message=command, pressed=pressed, last=time.monotonic(),
                             bumper=bool(message.buttons[deadman - 1]) if valid and deadman > 0 else False)

        ros.subscribe(Joy, "/joy", callback, image=True)
        driving = False
        try:
            while ros.ok():
                with lock:
                    pressed = state["pressed"] and time.monotonic() - state["last"] < .5
                    if pressed:
                        publisher.publish(state["message"])
                    elif driving:
                        publisher.publish(Twist())
                    driving = pressed
                    bumper_pub.publish(Bool(data=state["bumper"] if pressed else False))
                ros.sleep(1 / 9)
        finally:
            if driving:
                publisher.publish(Twist())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
