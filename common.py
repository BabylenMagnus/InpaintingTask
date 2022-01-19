from random import random, randint
import torch
from constant import IMAGE_SIZE, MASK_PERCENT, NUM_SQARES, SQUARE_PART


def get_square():
    unit = int(IMAGE_SIZE * MASK_PERCENT)
    x, y = randint(0, IMAGE_SIZE - unit),randint(0, IMAGE_SIZE - unit)
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
        x, y = randint(0, IMAGE_SIZE - one_square_side), randint(0, IMAGE_SIZE - one_square_side)
        mask[x:x + one_square_side, y:y + one_square_side] = 0

    return mask


def get_mask():
    return (get_square() if random() < .5 else get_multisquare()) \
        if random() < SQUARE_PART else get_noise()
