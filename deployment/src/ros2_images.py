"""RGB conversion respecting ROS 2 image encoding and row padding."""

from array import array

import numpy as np
from PIL import Image as PILImage
from sensor_msgs.msg import Image


def image_stamp_ns(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def msg_to_pil(message):
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    if message.encoding not in channels:
        raise ValueError(f"Unsupported RGB camera encoding: {message.encoding!r}")
    count = channels[message.encoding]
    row_bytes = message.width * count
    if message.width <= 0 or message.height <= 0 or message.step < row_bytes:
        raise ValueError("Invalid image dimensions or row stride")
    data = np.frombuffer(message.data, dtype=np.uint8)
    if data.size != message.height * message.step:
        raise ValueError("Image data length does not match height * step")
    pixels = data.reshape(message.height, message.step)[:, :row_bytes]
    pixels = pixels.reshape(message.height, message.width, count)
    if message.encoding in ("bgr8", "bgra8"):
        pixels = pixels[:, :, [2, 1, 0]]
    elif count == 4:
        pixels = pixels[:, :, :3]
    elif count == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    return PILImage.fromarray(pixels.copy(), mode="RGB")


def pil_to_msg(pil_image, stamp=None):
    pixels = np.asarray(pil_image.convert("RGB"))
    message = Image()
    message.encoding = "rgb8"
    message.height, message.width = pixels.shape[:2]
    message.step = message.width * 3
    message.data = array("B", pixels.tobytes())
    if stamp is not None:
        message.header.stamp = stamp
    return message
