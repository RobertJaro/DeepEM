import torch
from torch import nn


class DeepEM(nn.Module):

    def __init__(self, channels, t_bins, k, normalization):
        super().__init__()
        #
        self.d_logT = 0.05
        self.k = k  # (t_bins, channels)
        self.normalization = normalization  # (channels)
        #
        self.conv1 = nn.Conv2d(channels, 128, 3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv2d(512, 256, 3, padding=1, padding_mode='reflect')
        self.conv5 = nn.Conv2d(256, 128, 3, padding=1, padding_mode='reflect')
        self.conv6 = nn.Conv2d(128, t_bins, 3, padding=1, padding_mode='reflect')
        #

    def forward(self, x):
        # transform to DEM
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)
        x = self.conv6(x)
        x = torch.tanh(x)
        #
        log_dem = ((x + 1) / 2) * 3 + 26  # scale to [26, 29]
        dem = 10 ** log_dem  # compute dem(log T)
        y = torch.einsum('ijkl,jm->imkl', dem, self.k) * self.d_logT  # DN / s / px; fixed value range
        aia_normalized = y / self.normalization[None, :, None, None]  # scale to [0, 1]
        aia_normalized = (aia_normalized * 2) - 1 # scale [-1, 1]
        #
        return aia_normalized, log_dem


class TVLoss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def forward(self,x):
        return self.weight * torch.mean(torch.pow(x[:, 1:] - x[:, :-1], 2))