
from ros2_images import msg_to_pil, pil_to_msg

# pytorch
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
from typing import List

# Shared loader deliberately has no ROS or training/logging dependencies.
from vint_train.models.weights import load_model
from vint_train.data.data_utils import resize_and_aspect_crop


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Normalize and stack images; center_crop=True uses training's 4:3 crop/resize."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        if center_crop:
            image_tensor = resize_and_aspect_crop(pil_img, image_size)
        else:
            image_tensor = TF.to_tensor(pil_img.resize(image_size))
        transf_img = normalize(image_tensor)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)
    

# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi
