from common import *
import os
import cv2
import torch
from dataset import AFHQDataset


path = "data/afhq/train/cat"
imgs = os.listdir(path)

train_data = AFHQDataset("data/afhq/train")

for i in range(50):
    img, target, mask = train_data[i]
    img *= 256
    img = img.permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join('test', imgs[i]), img)

# for i in range(10):
#     img = cv2.imread(os.path.join(path, imgs[i]))
#     img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#
#     img = torch.from_numpy(img)
#     img = img.permute(2, 0, 1)
#
#     for name, func in (('multi', get_multisquare), ('noise', get_noise), ('square', get_square)):
#         mask = func()
#         imgn = img[:]
#         imgn = imgn * mask
#         # print(imgn.max(). imgn.min())
#         imgn = imgn.permute(1, 2, 0).numpy()
#         cv2.imwrite(os.path.join('test', name + imgs[i]), imgn)

