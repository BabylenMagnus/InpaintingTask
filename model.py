import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super(ConvolutionLayer, self).__init__()
        self.activation = nn.LeakyReLU(.2, inplace=True)
        if activation:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(.2, inplace=True)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)

        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.Sigmoid()
        )
        self.norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        mask = self.mask_conv(x)
        x = self.conv(x)

        return self.norm(x * mask)


class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownLayer, self).__init__()
        self.activation = nn.LeakyReLU(.2, inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(.2, inplace=True)
        )

        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.Sigmoid()
        )
        self.norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        mask = self.mask_conv(x)
        x = self.conv(x)

        return self.norm(x * mask)


class DilatedLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedLayer, self).__init__()
        self.activation = nn.LeakyReLU(.2, inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=dilation, dilation=dilation),
            nn.LeakyReLU(.2, inplace=True)
        )

        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=dilation, dilation=dilation),
            nn.Sigmoid()
        )
        self.norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        mask = self.mask_conv(x)
        x = self.conv(x)

        return self.norm(x * mask)


class UpLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpLayer, self).__init__()
        self.conv2d = ConvolutionLayer(in_channels, out_channels)

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)


class InpaitingModel(nn.Module):
    def __init__(self):
        super(InpaitingModel, self).__init__()

        n_channels = 16

        self.input = nn.Sequential(
            nn.Conv2d(4, n_channels, kernel_size=(5, 5), padding=(2, 2)),
            nn.LeakyReLU(inplace=True)
        )
        self.input_mask = nn.Sequential(
            nn.Conv2d(4, n_channels, kernel_size=(5, 5), padding=(2, 2)),
            nn.Sigmoid()
        )

        self.norm = nn.BatchNorm2d(n_channels)

        self.main = nn.Sequential(
            DownLayer(n_channels, 2 * n_channels),
            DownLayer(2 * n_channels, 4 * n_channels),

            DilatedLayer(4 * n_channels, 4 * n_channels, 2),
            DilatedLayer(4 * n_channels, 4 * n_channels, 4),

            UpLayer(4 * n_channels, 2 * n_channels),
            UpLayer(2 * n_channels, n_channels),
            ConvolutionLayer(n_channels, n_channels // 2),
            ConvolutionLayer(n_channels // 2, 3)
        )

    def forward(self, x):
        mask = self.input_mask(x)
        x = self.input(x)
        x = self.norm(x * mask)

        x = self.main(x)

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

    def forward(self, input):
        x = self.conv2d(input)
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

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0), -1))
        x = self.linear(x)
        return x
