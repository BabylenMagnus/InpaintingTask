import os
import cv2
from random import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

import numpy as np
from constant import *
from common import get_square, get_noise, get_multisquare


class InpaitingDataset(Dataset):
    def __init__(self, root='dataset/train', image_size=(256, 256)):
        self.root = root
        self.list_of_data = []
        self.list_of_target = []

        for file in os.listdir(self.root):
            if 'x' in file:
                self.list_of_data.append(file)
            if 'y' in file:
                self.list_of_target.append(file)

        self.image_size = image_size

    def __len__(self):
        return len(self.list_of_data)

    def __getitem__(self, item):
        name = self.list_of_data[item]
        path = os.path.join(self.root, name)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)

        mask = np.logical_and(
            img[:, :, 0] < dark_threshold, img[:, :, 1] < dark_threshold, img[:, :, 2] < dark_threshold
        ).astype(int)
        mask = F.to_tensor(mask)
        img = F.to_tensor(img)

        path_target = os.path.join(self.root, name.replace('x', 'y'))

        target = cv2.imread(path_target)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, self.image_size)
        target = F.to_tensor(target)

        return img, target, mask


class AFHQDataset(Dataset):
    def __init__(self, root='data/afhq/train'):
        self.root = root
        self.list_of_data = []

        for file in os.listdir(self.root):
            self.list_of_data += [os.path.join(file, x) for x in os.listdir(os.path.join(self.root, file))]

    def __len__(self):
        return len(self.list_of_data)

    def __getitem__(self, item):
        name = self.list_of_data[item]
        path = os.path.join(self.root, name)
        target = cv2.imread(path)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, (IMAGE_SIZE, IMAGE_SIZE))

        target = F.to_tensor(target)

        mask = (get_square() if random() < .5 else get_multisquare()) \
            if random() < SQUARE_PART else get_noise()

        img = target * mask
        mask.unsqueeze_(0)
        mask = mask.to(torch.float32)
        return img, target, mask
