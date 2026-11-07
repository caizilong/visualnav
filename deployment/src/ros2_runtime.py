"""ROS 2 execution and QoS shared by the deployment entry points.

Inference runs on the caller's thread. A separate executor keeps camera,
reset and stop callbacks responsive while diffusion is running. Wall-clock
waits deliberately keep watchdogs working when /clock is paused.
"""

import sys
import threading

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args


COMMAND_QOS = QoSProfile(depth=1)
STATE_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
IMAGE_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


def application_args():
    return remove_ros_args(args=sys.argv)[1:]


class ROS2Runtime:
    def __init__(self, name, args=None):
        self.context = Context()
        rclpy.init(args=args, context=self.context)
        self.node = rclpy.create_node(name, context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.stopped = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        try:
            while not self.stopped.is_set() and self.context.ok():
                self.executor.spin_once(timeout_sec=0.1)
        except Exception as exc:
            if not self.stopped.is_set() and self.context.ok():
                self.error = exc
        finally:
            self.stopped.set()

    def ok(self):
        if self.error is not None:
            raise RuntimeError("ROS 2 callback failed") from self.error
        return self.context.ok() and not self.stopped.is_set()

    def sleep(self, seconds):
        self.stopped.wait(seconds)

    def publisher(self, message_type, topic, *, state=False):
        return self.node.create_publisher(message_type, topic, STATE_QOS if state else COMMAND_QOS)

    def subscribe(self, message_type, topic, callback, *, state=False, image=False):
        qos = IMAGE_QOS if image else STATE_QOS if state else COMMAND_QOS
        return self.node.create_subscription(message_type, topic, callback, qos)

    def close(self):
        self.stopped.set()
        self.executor.shutdown(timeout_sec=2.)
        self.thread.join(timeout=2.)
        self.node.destroy_node()
        self.context.try_shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

