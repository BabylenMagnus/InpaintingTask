from random import random, randint
import torch
from constant import IMAGE_SIZE, NUM_SQARES


def get_center_square():
    side = randint(int(IMAGE_SIZE / 2.5), int(IMAGE_SIZE / 1.6))
    l = int(IMAGE_SIZE / 2.0 - side / 2.0)
    u = int(IMAGE_SIZE / 2.0 + side / 2.0)
    mask = torch.ones((1, IMAGE_SIZE, IMAGE_SIZE))
    mask[:, l:u, l:u] = 0
    return mask


def get_noise():
    r1, r2 = .5, .95
    ratio_to_throw = (r1 - r2) * torch.rand(1, 1, 1) + r2
    ratio_to_throw = ratio_to_throw.repeat(1, IMAGE_SIZE, IMAGE_SIZE)
    mask = (
            torch.rand(1, IMAGE_SIZE, IMAGE_SIZE) > ratio_to_throw
    ).to(int).to(torch.float32)
    return mask


def get_squares():
    ll_x = torch.randint(-IMAGE_SIZE * 2, IMAGE_SIZE * 3, (NUM_SQARES,))
    ll_y = torch.randint(-IMAGE_SIZE * 2, IMAGE_SIZE * 3, (NUM_SQARES,))
    sides = torch.randint(int(IMAGE_SIZE / 5), int(IMAGE_SIZE / 3), (NUM_SQARES,))
    ur_x = ll_x + sides
    ur_y = ll_y + sides

    mask = torch.ones((1, IMAGE_SIZE, IMAGE_SIZE))
    for xl, xu, yl, yu in zip(ll_x, ur_x, ll_y, ur_y):
        mask[:, xl:xu, yl:yu] = 0

    return mask


def get_mask():
    n = random()
    return get_noise() if n < .3 else get_squares() if n > .6 else get_center_square()
