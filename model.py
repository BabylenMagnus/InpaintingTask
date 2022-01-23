import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super(ConvBlock, self).__init__()
        self.a = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), dilation=(2, 2), padding=4)
        self.b = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), dilation=(5, 5), padding=10)
        self.c = nn.Conv2d(in_channels, out_channels * 2, kernel_size=(5, 5), padding=2)

        self.max_pool = None

        if pool:
            self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.max_pool:
            x = self.max_pool(x)

        x1 = F.leaky_relu(self.a(x))
        x2 = F.leaky_relu(self.b(x))
        x3 = F.leaky_relu(self.c(x))

        return torch.cat((x1, x2, x3), 1)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2).cuda()
        self.a = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(5, 5), dilation=2, padding=(4, 4))
        self.b = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(5, 5), dilation=5, padding=(10, 10))
        self.c = nn.ConvTranspose2d(in_channels, out_channels * 2, kernel_size=(5, 5), padding=(2, 2))

    def forward(self, x, y):
        x = self.up(x)
        x = torch.cat((x, y), 1)

        x1 = F.leaky_relu(self.a(x))
        x2 = F.leaky_relu(self.b(x))
        x3 = F.leaky_relu(self.c(x))

        return torch.cat((x1, x2, x3), 1)


class InpaitingModel(nn.Module):
    def __init__(self):
        super(InpaitingModel, self).__init__()

        self.conv1 = ConvBlock(7, 32).cuda()
        self.conv2 = ConvBlock(128, 32, pool=True).cuda()
        self.conv3 = ConvBlock(128, 64, pool=True).cuda()
        self.conv4 = ConvBlock(256, 128, pool=True).cuda()
        self.conv5 = ConvBlock(512, 128, pool=True).cuda()

        self.dec1 = DeconvBlock(1024, 64).cuda()
        self.dec2 = DeconvBlock(512, 32).cuda()
        self.dec3 = DeconvBlock(256, 32).cuda()
        self.dec4 = DeconvBlock(256, 32).cuda()

        self.out1 = nn.Sequential(
            nn.ConvTranspose2d(135, 8, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU()
        )

        self.out2 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=(3, 3), padding=(1, 1)),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x6 = self.dec1(x5, x4)
        x7 = self.dec2(x6, x3)
        x8 = self.dec3(x7, x2)
        x9 = self.dec4(x8, x1)

        x = torch.cat((x9, x), 1)
        x = self.out1(x)
        x = self.out2(x)

        return x


class SpectralDownLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpectralDownLayer, self).__init__()
        self.conv2d = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv2d(x)
        return x


class InpaintingDiscriminator(nn.Module):
    def __init__(self):
        super(InpaintingDiscriminator, self).__init__()
        cnum = 16
        self.discriminator_net = nn.Sequential(
            SpectralDownLayer(3, cnum),
            SpectralDownLayer(cnum, 2 * cnum),
            SpectralDownLayer(2 * cnum, 4 * cnum),
            SpectralDownLayer(4 * cnum, 8 * cnum),
            SpectralDownLayer(8 * cnum, 8 * cnum),
            SpectralDownLayer(8 * cnum, 8 * cnum),
            SpectralDownLayer(8 * cnum, 8 * cnum)
        )
        self.linear = nn.Sequential(
            nn.Linear(8 * cnum * 2 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discriminator_net(x)
        x = x.view((x.size(0), -1))
        x = self.linear(x)
        return x
