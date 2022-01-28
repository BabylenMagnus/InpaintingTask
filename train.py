from engine import train_one_epoch, get_generator, get_discriminator
from dataset import AFHQDataset

from torch.utils.data import DataLoader
from torch.optim import Adam
import torch

import sys
from datetime import datetime

from constant import *


time = datetime.now().strftime("%d_%H_%m")

train_data = AFHQDataset("data/afhq/train/cat")
test_data = AFHQDataset("data/afhq/val/cat")

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=12
)

test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE, num_workers=12
)

generator = get_generator().cuda()
discriminator = get_discriminator().cuda()

optim_gen = Adam(generator.parameters(), lr=LR, betas=(0.5, 0.9))
optim_dis = Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.9))


def main(epoch):
    for i in range(1, epoch + 1):
        gen_loss, dis_loss = train_one_epoch(
            generator, discriminator, optim_gen, optim_dis, train_loader
        )
        print(f"epoch: {i}; train generator loss: {round(gen_loss, 4)}; train discriminator loss: {round(dis_loss, 4)}")

        if not i % TEST_RATE:
            test_gen_loss, test_dis_loss = 0, 0
            with torch.no_grad():
                for imgs, targets, masks in test_loader:
                    optim_gen.zero_grad()

                    targets = targets.cuda()

                    mask3 = torch.cat((masks, masks, masks), 1)
                    rand = torch.randn(mask3.shape)
                    rand = rand * (1 - mask3)
                    inp_tensor = torch.cat((imgs, masks, rand), dim=1).cuda()

                    optim_dis.zero_grad()
                    gen_img = generator(inp_tensor)

                    real_validity = discriminator(targets).reshape(-1)
                    fake_validity = discriminator(gen_img).reshape(-1)

                    # Wasserstein loss
                    disc_loss = -(torch.mean(real_validity) - torch.mean(fake_validity))
                    test_dis_loss += disc_loss.cpu().item()

                    optim_gen.zero_grad()

                    gen_img = generator(inp_tensor)
                    fake_validity = discriminator(gen_img)
                    g_loss = -torch.mean(fake_validity)

                    test_gen_loss += g_loss.cpu().item()

            test_gen_loss /= len(test_loader)
            test_dis_loss /= len(test_loader)

            print(
                f"epoch: {i}; test generator loss: {round(test_gen_loss, 4)}; test discriminator loss: "
                f"{round(test_dis_loss, 4)}"
            )

        if not i % SAVE_RATE:
            gen_path = f'wgen_{time}_{i}.pt'
            dis_name = f'wdis_{time}_{i}.pt'

            torch.save(generator.state_dict(), "weights/" + gen_path)
            torch.save(discriminator.state_dict(), "weights/" + dis_name)


if __name__ == '__main__':
    main(int(sys.argv[1]))
