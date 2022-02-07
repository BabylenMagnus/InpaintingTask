import os
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

import numpy as np
from constant import *
from common import get_mask


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
        self.list_of_data = os.listdir(self.root)

    def __len__(self):
        return len(self.list_of_data)

    def __getitem__(self, item):
        name = self.list_of_data[item]
        path = os.path.join(self.root, name)
        x = cv2.imread(path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))

        x = F.to_tensor(x)

        mask = get_mask()

        new_x = x * mask
        mask3 = mask.repeat(3, 1, 1)
        rand_z = torch.distributions.normal.Normal(0, 0.1).sample(mask3.shape)
        rand_z = (1 - mask3) * rand_z

        mask = mask.to(torch.float32)

        return x, new_x, mask, rand_z
