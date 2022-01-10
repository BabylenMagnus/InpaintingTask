from engine import train_one_epoch, get_generator, get_discriminator
from dataset import InpaitingDataset

from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch

import sys
from datetime import datetime

from constant import *
import os
from random import choice
import cv2
from torchvision.transforms import functional as F

import numpy as np
import matplotlib.pyplot as plt


time = datetime.now().strftime("%d_%H_%m")

train_data = InpaitingDataset("dataset/train")
test_data = InpaitingDataset("dataset/test")

train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True, num_workers=12
)

test_loader = DataLoader(
    test_data, batch_size=16, num_workers=12
)

print(len(train_loader))

generator = get_generator().cuda()
discriminator = get_discriminator().cuda()

optim_gen = Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
optim_dis = Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

epoch = int(sys.argv[1])

gen_loss_func = nn.MSELoss()
dis_loss_func = nn.BCELoss()

train_generator = True

for i in range(1, epoch + 1):
    gen_loss, dis_loss = train_one_epoch(
        generator, discriminator, gen_loss_func, dis_loss_func, optim_gen, optim_dis, train_loader, train_generator
    )

    train_generator = dis_loss < .8

    print(f"epoch: {i}; train generator loss: {round(gen_loss, 4)}; train discriminator loss: {round(dis_loss, 4)}")

    if not i % 20:
        test_gen_loss, test_dis_loss = 0, 0
        with torch.no_grad():
            for _, (imgs, targets, masks) in enumerate(test_loader):
                optim_gen.zero_grad()
                optim_dis.zero_grad()

                batch_size = len(imgs)
                inp_tensor = torch.cat((imgs, masks), dim=1).cuda()

                # generator forward
                gen_img = generator(inp_tensor)

                # discriminator forward
                batch_data = torch.cat([targets.cuda(), gen_img.detach()], dim=0)
                batch_output = discriminator(batch_data)
                real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

                # discriminator loss
                zero_classes = torch.zeros_like(fake_pred).cuda()
                ones_classes = torch.ones_like(real_pred).cuda()
                dis_loss = dis_loss_func(fake_pred, zero_classes) + dis_loss_func(real_pred, ones_classes)

                test_dis_loss += dis_loss.cpu().item()

                # generator loss
                targets = targets.cuda()
                masks = masks.cuda()

                fake_pred = discriminator(gen_img)
                gen_loss = \
                    gen_loss_func(targets, gen_img) \
                    + (1 - torch.mean(fake_pred)) * alpha \
                    + gen_loss_func(targets * masks, gen_img * masks) * mask_alpha

                test_gen_loss += gen_loss.cpu().item()

        test_gen_loss /= len(test_loader)
        test_dis_loss /= len(test_loader)

        print(
            f"epoch: {i}; test generator loss: {round(test_gen_loss, 4)}; test discriminator loss: "
            f"{round(test_dis_loss, 4)}"
        )

    if not i % 50:
        gen_path = f'gen_{time}_{i}.pt'
        dis_name = f'dis_{time}_{i}.pt'

        torch.save(generator.state_dict(), "weights/" + gen_path)
        torch.save(discriminator.state_dict(), "weights/" + dis_name)

        val_data = os.listdir(val_data_path)
        val_data = list(filter(lambda x: "x" in x, val_data))

        name = choice(val_data)

        img = os.path.join(val_data_path, name)

        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        mask = img == [0, 0, 0]
        mask = F.to_tensor(mask)
        mask = mask[0]
        mask = mask.unsqueeze(0)
        mask = mask.to(torch.float32)

        img = F.to_tensor(img)

        inp_tensor = torch.cat((img, mask), dim=0).unsqueeze(0).cuda()
        out = generator(inp_tensor)[0]
        out = out.detach().cpu().numpy()
        out = np.transpose(out, (1, 2, 0))

        out[out < 0] = 0
        out[out > 1] = 1

        plt.imsave(os.path.join(output_path, f"{time}_{i}.jpg"), out)
