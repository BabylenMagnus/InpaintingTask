import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

from engine import get_generator
from common import get_mask
from constant import IMAGE_SIZE


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
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = F.to_tensor(img)

    mask = get_mask()
    img = img * mask

    img_ = img.numpy()
    img_ = img_ * 256
    img_ = np.int16(img_)
    img_ = np.transpose(img_, (1, 2, 0))
    img_ = img_[:, :, [2, 1, 0]]
    cv2.imwrite(os.path.join(args.output_folder_path, "t" + i), img_)

    mask.unsqueeze_(0)
    mask3 = torch.cat((mask, mask, mask), 0)
    rand = torch.randn(mask3.shape)
    rand = rand * (1 - mask3)
    inp_tensor = torch.cat((img, mask, rand), dim=0).unsqueeze(0).cuda()
    gen_img = generator(inp_tensor)[0].detach().cpu()
    img = gen_img * (1 - mask3) + img

    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))

    img[img < 0] = 0
    img[img > 1] = 1

    img *= 256
    img = np.int16(img)
    img = img[:, :, [2, 1, 0]]

    cv2.imwrite(os.path.join(args.output_folder_path, i), img)
