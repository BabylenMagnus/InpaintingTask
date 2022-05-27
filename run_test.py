import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from engine import get_generator
from common import get_mask
from constant import IMAGE_SIZE

parser = ArgumentParser()
parser.add_argument('--test_folder_path', type=str)
parser.add_argument('--output_folder_path', type=str)
args = parser.parse_args()

generator = get_generator().cuda()
generator.eval()


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


for i in os.listdir(args.test_folder_path):
    path = os.path.join(args.test_folder_path, i)

    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))

    x = F.to_tensor(x)

    mask = get_mask()

    new_x = x * mask
    print(new_x)
    show(new_x)
    plt.show()
    # show(mask)
    print(type(new_x), new_x.shape)

    mask3 = mask.repeat(3, 1, 1)
    rand_z = torch.distributions.normal.Normal(0, 0.1).sample(mask3.shape)
    rand_z = (1 - mask3) * rand_z

    mask = mask.to(torch.float32)

    inp_tensor = torch.cat((new_x, mask, rand_z), dim=0).unsqueeze(0).cuda()
    gen_img = generator(inp_tensor)[0].detach().cpu()
    img = gen_img * (1 - mask3) + new_x

    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    print(img)
    img[img < 0] = 0
    img[img > 1] = 1

    img *= 256
    img = np.int16(img)
    img = img[:, :, [2, 1, 0]]

    cv2.imwrite(os.path.join(args.output_folder_path, i), img)
