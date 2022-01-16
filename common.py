import random
import torch


def get_mask(img, percent=.9):
    size = img.shape[-1]
    unit = int(size * percent)
    w, h = random.randrange(0, size - unit), random.randrange(unit, size)
    mask = torch.zeros_like(img)
    mask[:, h - unit: h, w: w + unit] += torch.ones(unit, unit)
    return mask
