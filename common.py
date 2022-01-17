import random
import torch
from constant import *


def get_square():
    unit = int(IMAGE_SIZE * MASK_PERCENT)
    w, h = random.randrange(0, IMAGE_SIZE - unit), random.randrange(unit, IMAGE_SIZE)
    mask = torch.ones(IMAGE_SIZE, IMAGE_SIZE)
    mask[h - unit: h, w: w + unit] -= torch.ones(unit, unit)
    return mask


def get_noise():
    mask = torch.rand(IMAGE_SIZE, IMAGE_SIZE)
    mask = mask > MASK_PERCENT
    mask = mask.to(int)
    return mask
