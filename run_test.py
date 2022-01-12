import os
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import functional as F

from engine import get_generator
from constant import dark_threshold


parser = ArgumentParser()
parser.add_argument('--test_folder_path', type=str)
parser.add_argument('--output_folder_path', type=str)
args = parser.parse_args()

generator = get_generator().cuda()
generator.eval()


for i in os.listdir(args.test_folder_path):
    path = os.path.join(args.test_folder_path, i)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    mask = np.logical_and(
        img[:, :, 0] < dark_threshold, img[:, :, 1] < dark_threshold, img[:, :, 2] < dark_threshold
    ).astype(int)

    mask = F.to_tensor(mask)
    # mask = mask.to(torch.float32)

    img = F.to_tensor(img)

    inp_tensor = torch.cat((img, mask), dim=0).unsqueeze(0).cuda()
    out = generator(inp_tensor)[0]

    mask = torch.cat((mask, mask, mask)).to(bool)
    out = out.cpu()
    img[mask] = out[mask]

    img = img.detach().numpy()
    img = np.transpose(img, (1, 2, 0))

    img[img < 0] = 0
    img[img > 1] = 1

    plt.imsave(os.path.join(args.output_folder_path, i), img)
