import torch
import os
from model import InpaitingModel, InpaintingDiscriminator
from constant import *


def compute_gp(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    # Sample Epsilon from uniform distribution
    eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
    eps = eps.expand_as(real_data)

    # Interpolation between real data and fake data.
    interpolation = eps * real_data + (1 - eps) * fake_data

    # get logits for interpolated images
    interp_logits = netD(interpolation)
    grad_outputs = torch.ones_like(interp_logits)

    # Compute Gradients
    gradients = torch.autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute and return Gradient Norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)


def train_one_epoch(
        generator, discriminator, optim_gen, optim_dis, dataloader
):
    generator.train()
    discriminator.train()

    all_gen_loss, all_dis_loss = 0, 0

    mse_loss = torch.nn.MSELoss()

    for i, (imgs, targets, masks) in enumerate(dataloader):
        targets = targets.cuda()
        inp_tensor = torch.cat((imgs, masks), dim=1).cuda()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optim_dis.zero_grad()
        gen_img = generator(inp_tensor)

        real_validity = discriminator(targets).reshape(-1)
        fake_validity = discriminator(gen_img).reshape(-1)

        # Gradient Penalty
        gp = compute_gp(discriminator, targets, gen_img)

        # Wasserstein loss with penalty
        d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + penalty_lambda * gp
        d_loss.backward()
        optim_dis.step()

        all_dis_loss += d_loss.detach().cpu().item()

        if not i % steps_generator_train:
            # -----------------
            #  Train Generator
            # -----------------
            optim_gen.zero_grad()

            gen_img = generator(inp_tensor)
            fake_validity = discriminator(gen_img).reshape(-1)
            g_loss = -torch.mean(fake_validity) + mse_loss(gen_img, targets) * mse_alpha

            g_loss.backward()
            optim_gen.step()

            all_gen_loss += g_loss.detach().cpu().item()

    all_gen_loss /= len(dataloader)
    all_dis_loss /= len(dataloader)

    return all_gen_loss, all_dis_loss


def get_generator():
    weights = sorted(os.listdir("weights"))
    weights = list(filter(lambda x: x.startswith("wgen"), weights))

    generator = InpaitingModel().cuda()

    if weights:
        generator.load_state_dict(torch.load("weights/" + weights[-1]))

    return generator


def get_discriminator():
    weights = sorted(os.listdir("weights"))
    weights = list(filter(lambda x: x.startswith("wdis"), weights))

    discriminator = InpaintingDiscriminator().cuda()

    if weights:
        discriminator.load_state_dict(torch.load("weights/" + weights[-1]))

    return discriminator
