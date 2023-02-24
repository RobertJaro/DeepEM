import torch
from torch import nn


class DEMModel(nn.Module):

    def __init__(self, channels, n_normal, T, k, normalization, n_dims=512, scaling_factor=1e22):
        super().__init__()
        #
        self.register_buffer("k", k)
        self.register_buffer("scaling_factor", torch.tensor(scaling_factor, dtype=torch.float32))
        self.register_buffer("normalization", normalization)
        self.register_buffer("stretch_div", torch.arcsinh(torch.tensor(1 / 0.005)))

        self.register_buffer("T", T)
        self.register_buffer("dT", torch.gradient(T)[0])
        #
        convs = []
        # channels + error
        convs += [nn.Conv2d(channels * 2, n_dims, 1, padding=0, padding_mode='reflect'), Sine(), ]
        for _ in range(4):
            convs += [nn.Conv2d(n_dims, n_dims, 1, padding=0, padding_mode='reflect'), Sine(), ]
        # dropout layer
        convs += [nn.Conv2d(n_dims, n_dims, 1, padding=0, padding_mode='reflect'), Sine(), nn.Dropout2d(0.1), ]
        self.convs = nn.Sequential(*convs)
        self.out = nn.Conv2d(n_dims, 3 * n_normal, 1, padding=0, padding_mode='reflect')
        #
        self.out_act = nn.Softplus()

    def forward(self, x, T=None, return_em = False):
        if T is not None:
            T = T  # default log T
            dT = torch.gradient(T)[0]
        else:
            T = self.T
            dT = self.dT
        #
        dem = self._compute_dem(x, T)
        em = dem * dT[None, :, None, None]
        euv_normalized = self.compute_euv(em)
        #
        dem = dem * self.scaling_factor  # compute DEM from EM (use correct T bins)
        #
        if return_em:
            return euv_normalized, dem, em
        return euv_normalized, dem

    def compute_euv(self, em):
        y = torch.einsum('ijkl,jm->imkl', em, self.k * self.scaling_factor)  # 1e30 DN / s / px
        euv_normalized = y / self.normalization[None, :, None, None]  # scale to [0, 1]
        euv_normalized = torch.true_divide(torch.arcsinh(euv_normalized / 0.005), self.stretch_div)  # stretch
        euv_normalized = (euv_normalized * 2) - 1  # scale approx. [-1, 1]
        # real DEM = EM / [K bin] = dem * dlogT / [K bin]
        return euv_normalized

    def _compute_dem(self, x, T):
        # scale to Mm
        T = T[None, None, :, None, None] * 1e-6  # (batch, n_normal, T_bins, w, h)
        # transform to DEM
        x = self.convs(x)
        x = self.out(x) # [0, 2e22]
        x = x.view(x.shape[0], -1, 3, *x.shape[-2:]) # x (batch, n_normal, 3, w, h)
        # min width of 1 temperature bin
        std = self.out_act(x[:, :, 0, None, :, :]) * (20 - 1) * 1e-2 + 1e-2 # [.01, .2] MK
        mean = torch.sigmoid(x[:, :, 1, None, :, :]) * (T.max() - T.min()) + T.min()
        # print(std.min(), std.max())
        # mean = torch.linspace(logT.min() + 0.1, logT.max() - 0.1, x.shape[1], dtype=torch.float32, device=x.device)[None, :, None, None, None]
        w = torch.sigmoid(x[:, :, 2, None, :, :]) # [0, 2]
        normal = w * torch.exp(-0.5 * ((T - mean) / (std + 1e-8)) ** 2)
        dem = normal.sum(1)
        # print('1:', mean[0, 0, 0, 0, 0].detach().cpu().numpy(), std[0, 0, 0, 0, 0].detach().cpu().numpy(), w[0, 0, 0, 0, 0].detach().cpu().numpy())
        # print('2:', mean[0, 1, 0, 0, 0].detach().cpu().numpy(), std[0, 1, 0, 0, 0].detach().cpu().numpy(), w[0, 1, 0, 0, 0].detach().cpu().numpy())
        # print('3:', mean[0, 2, 0, 0, 0].detach().cpu().numpy(), std[0, 2, 0, 0, 0].detach().cpu().numpy(), w[0, 2, 0, 0, 0].detach().cpu().numpy())
        return dem


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
