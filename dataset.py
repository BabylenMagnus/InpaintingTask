import os
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import torch
import numpy as np


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
        img = cv2.resize(img, (256, 256))

        mask = img == [0, 0, 0]
        mask = F.to_tensor(mask)
        mask = mask[0]
        mask = mask.unsqueeze(0)
        mask = mask.to(torch.float32)

        img = F.to_tensor(img)

        path_target = os.path.join(self.root, name.replace('x', 'y'))

        target = cv2.imread(path_target)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, self.image_size)
        target = F.to_tensor(target)

        return img, target, mask
