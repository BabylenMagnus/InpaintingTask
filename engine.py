import torch
import os
from model import InpaitingModel, InpaintingDiscriminator
from constant import *


def train_one_epoch(
        generator, discriminator, gen_loss_func, dis_loss_func, optim_gen, optim_dis, dataloader, train_generator=False
):
    generator.train()
    discriminator.train()

    all_gen_loss, all_dis_loss = 0, 0

    for _, (imgs, targets, masks) in enumerate(dataloader):
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

        dis_loss.backward()
        optim_dis.step()

        all_dis_loss += dis_loss.cpu().item()

        # generator loss
        if train_generator:
            targets = targets.cuda()
            masks = masks.cuda()

            fake_pred = discriminator(gen_img)
            gen_loss = \
                gen_loss_func(targets, gen_img) \
                + (1 - torch.mean(fake_pred)) * alpha \
                + gen_loss_func(targets * masks, gen_img * masks) * mask_alpha

            gen_loss.backward()
            optim_gen.step()

            all_gen_loss += gen_loss.cpu().item()

    all_gen_loss /= len(dataloader)
    all_dis_loss /= len(dataloader)

    return all_gen_loss, all_dis_loss


def get_generator():
    weights = sorted(os.listdir("weights"))
    weights = list(filter(lambda x: x.startswith("gen"), weights))

    generator = InpaitingModel().cuda()

    if weights:
        generator.load_state_dict(torch.load("weights/" + weights[-1]))

    return generator


def get_discriminator():
    weights = sorted(os.listdir("weights"))
    weights = list(filter(lambda x: x.startswith("dis"), weights))

    discriminator = InpaintingDiscriminator().cuda()

    if weights:
        discriminator.load_state_dict(torch.load("weights/" + weights[-1]))

    return discriminator
