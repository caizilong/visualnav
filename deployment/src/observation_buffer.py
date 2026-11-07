"""Thread-safe RGB context with fresh ROS 2 timestamps and training stride."""

from collections import deque
import threading
import time

from ros2_images import image_stamp_ns, msg_to_pil


class ObservationBuffer:
    def __init__(self, context_size, stride=1, stale_timeout_s=2., clock=time.monotonic):
        self.frames = deque(maxlen=context_size + 1)
        self.stride, self.timeout, self.clock = stride, stale_timeout_s, clock
        self.lock = threading.RLock()
        self.last_stamp = 0
        self.last_raw_time = float("-inf")
        self.frame_id = 0
        self.sampled_id = 0

    def callback(self, message):
        stamp = image_stamp_ns(message)
        with self.lock:
            if stamp <= self.last_stamp:
                return  # ROS 2 has no header.seq; require a positive increasing stamp.
            image = msg_to_pil(message)
            if self.clock() - self.last_raw_time >= self.timeout:
                self.frames.clear()
                self.frame_id = 0
            self.last_stamp, self.last_raw_time = stamp, self.clock()
            self.frame_id += 1
            if (self.frame_id - 1) % self.stride == 0:
                self.frames.append(image)
                self.sampled_id += 1

    def fresh(self):
        with self.lock:
            return self.clock() - self.last_raw_time < self.timeout

    def snapshot(self):
        with self.lock:
            ready = len(self.frames) == self.frames.maxlen and self.fresh()
            return (self.sampled_id, list(self.frames)) if ready else (None, None)
