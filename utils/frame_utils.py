import numpy as np
import cv2
from os.path import *
import imageio
from . import flow_utils 

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imageio.imread(file_name)
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    return []

def resize_frame(image, target_height=256, target_width=256):
    """
    Only for target_height == target width. 256 is recommended.
    We suppose image is obtained by cv2.
    """
    if len(image) == 2:
        image = np.concatenate([image[..., None] for i in range(3)])
    elif len(image) == 4:
        image = image[..., 0]
    H, W, C = (*image.shape, )

    if H == W:
        image = cv2.resize(image, (target_height, target_width))
    elif H < W:
        image = cv2.resize(image, (int(W * target_height / H), target_height))
        new_W = image.shape[1]
        crop_len = new_W - target_width
        image = image[:, crop_len // 2: -(crop_len - crop_len // 2), :]
    else:
        image = cv2.resize(image, (target_width, int(H * target_width / W)))
        new_H = image.shape[0]
        crop_len = new_H - target_height
        image = image[crop_len // 2: -(crop_len - crop_len // 2), ...]

    return image

