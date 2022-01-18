import random
import torch
from constant import *


def get_square():
    unit = int(IMAGE_SIZE * MASK_PERCENT)
    x, y = random.randint(0, IMAGE_SIZE - unit), random.randint(0, IMAGE_SIZE - unit)
    mask = torch.ones(IMAGE_SIZE, IMAGE_SIZE)
    mask[y: y + unit, x: x + unit] = 0
    return mask


def get_noise():
    mask = torch.rand(IMAGE_SIZE, IMAGE_SIZE)
    mask = mask > MASK_PERCENT
    mask = mask.to(int)
    return mask


def get_multisquare():
    image_area = IMAGE_SIZE * IMAGE_SIZE
    damage_area = MASK_PERCENT * image_area
    one_square_area = damage_area / NUM_SQARES
    one_square_side = int(one_square_area ** (1 / 2))
    mask = torch.ones((IMAGE_SIZE, IMAGE_SIZE))

    for k in range(NUM_SQARES):
        x, y = random.randint(0, IMAGE_SIZE - one_square_side), random.randint(0, IMAGE_SIZE - one_square_side)
        mask[x:x + one_square_side, y:y + one_square_side] = 0

    return mask
