from functools import partial

import cv2
import torch
import numpy as np


def tensor2img(image_tensor, imtype=np.uint8, denormalize_fn=None, scale_factor=255.0):
    shape = image_tensor.shape
    image_tensor = image_tensor.cpu()
    if denormalize_fn is not None:
        image_tensor = denormalize_fn(image_tensor)
    np_img = image_tensor.float().numpy()
    if len(shape) == 4:
        ret = []
        for img in np_img:
            if img.shape[0] == 1:
                img = np.tile(img, (3, 1, 1))
            ret.append(img)
        ret = np.transpose(np.array(ret), (0, 2, 3, 1))
    elif len(shape) == 3:
        if shape[0] == 1:
            np_img = np.tile(np_img, (3, 1, 1))
        ret = np.transpose(np_img, (1, 2, 0))
    elif len(shape) == 2:
        ret = np_img
    else:
        print("Expected a Tensor of C x H x W or B x C x H x W")
        return
    return (ret * scale_factor).astype(imtype)
