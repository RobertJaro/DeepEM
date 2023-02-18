import numpy as np
import torch
from torch import nn


class DEMModel(nn.Module):

    def __init__(self, channels, n_normal, logT, k, normalization, n_dims=512, scaling_factor=1e29):
        super().__init__()
        #
        self.register_buffer("k", k)
        self.register_buffer("scaling_factor", torch.tensor(scaling_factor, dtype=torch.float32))
        self.register_buffer("normalization", normalization)
        self.register_buffer("stretch_div", torch.arcsinh(torch.tensor(1 / 0.005)))

        self.register_buffer("logT", logT)
        self.register_buffer("dT", torch.gradient(10 ** logT)[0])
        self.register_buffer("dlogT", torch.gradient(logT)[0])
        #
        convs = []
        convs += [nn.Conv2d(channels, n_dims, 1, padding=0, padding_mode='reflect'), Sine(), ]
        for _ in range(4):
            convs += [nn.Conv2d(n_dims, n_dims, 1, padding=0, padding_mode='reflect'), Sine(), ]
        # dropout layer
        convs += [nn.Conv2d(n_dims, n_dims, 1, padding=0, padding_mode='reflect'), Sine(), nn.Dropout2d(0.2), ]
        self.convs = nn.Sequential(*convs)
        self.out = nn.Conv2d(n_dims, n_normal * 3, 1, padding=0, padding_mode='reflect')
        #
        self.out_act = nn.Softplus()

    def forward(self, x, logT=None):
        if logT is not None:
            logT = logT  # default log T
            dlogT = torch.gradient(logT)[0]
        else:
            logT = self.logT
            dlogT = self.dlogT
        #
        dem = self._compute_dem(x, logT)
        em = dem * dlogT[None, :, None, None]
        euv_normalized = self.compute_euv(em)
        #
        dem = em / self.dT[None, :, None, None] * self.scaling_factor  # compute DEM from EM (use correct T bins)
        #
        return euv_normalized, dem

    def compute_euv(self, em):
        y = torch.einsum('ijkl,jm->imkl', em, self.k * self.scaling_factor)  # 1e30 DN / s / px
        euv_normalized = y / self.normalization[None, :, None, None]  # scale to [0, 1]
        euv_normalized = torch.true_divide(torch.arcsinh(euv_normalized / 0.005), self.stretch_div)  # stretch
        euv_normalized = (euv_normalized * 2) - 1  # scale approx. [-1, 1]
        # real DEM = EM / [K bin] = dem * dlogT / [K bin]
        return euv_normalized

    def _compute_dem(self, x, logT):
        dlogT = logT[1] - logT[0]
        logT = logT[None, None, :, None, None]  # (batch, n_normal, T_bins, w, h)
        # transform to DEM
        x = self.convs(x)
        x = self.out(x)
        x = x.view(x.shape[0], -1, 3, *x.shape[2:])
        # (batch, n_normal, T_bins, w, h)
        # min width of 1 temperature bin
        std = self.out_act(x[:, :, 0, None, :, :]) * 10 * torch.gradient(logT, dim=2)[0]
        mean = torch.sigmoid(x[:, :, 1, None, :, :]) * (self.logT.max() - self.logT.min()) + self.logT.min()
        # mean = torch.linspace(logT.min() + 0.1, logT.max() - 0.1, x.shape[1], dtype=torch.float32, device=x.device)[None, :, None, None, None]
        w = self.out_act(x[:, :, 2, None, :, :])
        normal = w * (std * np.sqrt(2 * np.pi) + 1e-8) ** -1 * torch.exp(-0.5 * (logT - mean) ** 2 / (std ** 2 + 1e-8))
        dem = normal.sum(1)
        return dem


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
