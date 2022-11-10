import os
from pathlib import Path
from urllib import request

import numpy as np
import torch
from astropy.nddata import block_reduce
from skimage.util import view_as_blocks
from torch import nn
from torch.utils.data import DataLoader

from dem.train.generator import DEMDataset


class DeepEMModel(nn.Module):

    def __init__(self, channels, n_normal, log_T, k, normalization):
        super().__init__()
        #
        self.register_buffer("k", k)
        self.register_buffer("normalization", normalization)
        self.register_buffer("d_logT", torch.tensor(0.05, dtype=torch.float32))
        self.register_buffer("logT", torch.tensor(log_T, dtype=torch.float32))
        #
        convs = []
        convs += [nn.Conv2d(channels, 128, 3, padding=1, padding_mode='reflect')]
        convs += [nn.Conv2d(128, 256, 3, padding=1, padding_mode='reflect')]
        convs += [nn.Conv2d(256, 512, 3, padding=1, padding_mode='reflect')]
        convs += [nn.Conv2d(512, 1024, 3, padding=1, padding_mode='reflect')]
        convs += [nn.Conv2d(1024, 1024, 3, padding=1, padding_mode='reflect')]
        convs += [nn.Conv2d(1024, 1024, 3, padding=1, padding_mode='reflect')]
        self.convs = nn.ModuleList(convs)
        self.out = nn.Conv2d(1024, n_normal * 3, 3, padding=1, padding_mode='reflect')
        #
        self.out_act = nn.Softplus()

    def forward(self, x):
        # transform to DEM
        for conv in self.convs:
            x = torch.sin(conv(x))
        x = self.out(x)
        x = x.view(x.shape[0], -1, 3, *x.shape[2:])
        std = self.out_act(x[:, :, 0, None, :, :]) # (batch, n_normal, T_bins, w, h)
        mean = torch.sigmoid(x[:, :, 1, None, :, :]) * (self.logT.max() - self.logT.min()) + self.logT.min()
        w = self.out_act(x[:, :, 2, None, :, :])
        logT = self.logT[None, None, :, None, None] # (batch, n_normal, T_bins, w, h)
        normal = w * (std * np.sqrt(2 * np.pi) + 1e-8) ** -1 * torch.exp(-0.5 * (logT - mean) ** 2 / (std ** 2 + 1e-8))
        log_dem = normal.sum(1)
        dem = 10 ** log_dem
        # print('mean', mean.min(), mean.max(), mean.mean())
        # print('std', std.min(), std.max())
        # print('weight', w.min(), w.max())
        # print('mag', mag.min(), mag.max())
        # print('dem', dem.min(), dem.max())
        #
        # log_dem = x + 28  # scale to approx [26, 29]
        # dem = 10 ** log_dem  # compute dem(log T)
        y = torch.einsum('ijkl,jm->imkl', dem, self.k * 1e26) * self.d_logT  # 1e26 DN / s / px
        aia_normalized = y / self.normalization[None, :, None, None]  # scale to [0, 1]
        aia_normalized = (aia_normalized * 2) - 1  # scale [-1, 1]
        #
        return aia_normalized, log_dem


class DeepEM:

    def __init__(self, model_path=None, model_name='deepem_v0_1.pt', device=None):
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu') if not device else device
        self.model = torch.load(model_path, map_location=device) if model_path else \
            torch.load(self._get_model_path(model_name), map_location=device)
        self.device = device
        self.log_T = np.linspace(4, 9, 101)

    def compute(self, image, return_reconstruction=False, bin=None, temperature_range=None):
        with torch.no_grad():
            image = torch.tensor(image, dtype=torch.float32).to(self.device)
            image = image[None,]  # expand batch dimension
            reconstruction, log_dem = self.model(image)
            reconstruction = reconstruction.cpu().detach().numpy()[0]
            log_dem = log_dem.detach().cpu().numpy()[0]
            if bin:
                log_dem = block_reduce(log_dem, bin)
                reconstruction = block_reduce(reconstruction, bin)
            if temperature_range:
                cond = np.ones_like(self.log_T, dtype=np.bool)
                if temperature_range[0] is not None:
                    cond = cond & (self.log_T >= temperature_range[0])
                if temperature_range[1] is not None:
                    cond = cond & (self.log_T <= temperature_range[1])
                log_dem = log_dem[cond]
            if return_reconstruction:
                return log_dem, reconstruction
            else:
                return log_dem

    def icompute(self, files, num_workers=4, block_shape=(512, 512), **kwargs):
        dataset = DEMDataset(files)
        loader = DataLoader(dataset, batch_size=1, num_workers=num_workers)
        for image in loader:
            if block_shape:
                yield self.compute_patches(image[0].numpy(), block_shape, **kwargs)
            else:
                yield self.compute(image, **kwargs)

    def compute_patches(self, img, block_shape, return_reconstruction=False, **kwargs):
        patch_shape = (img.shape[0], *block_shape)
        patches = view_as_blocks(img, patch_shape)
        patches = np.reshape(patches, (-1, *patch_shape))
        dem_patches = []
        rec_patches = []
        with torch.no_grad():
            for patch in patches:
                dem_patch, rec_patch = self.compute(patch, True, **kwargs)
                dem_patches.append(dem_patch)
                rec_patches.append(rec_patch)
        #
        dem_patches = np.array(dem_patches)
        dem_patches = dem_patches.reshape((img.shape[1] // block_shape[0], img.shape[2] // block_shape[1],
                                           dem_patches.shape[1], dem_patches.shape[2], dem_patches.shape[3]))
        dem = np.moveaxis(dem_patches, [0, 1], [1, 3]).reshape((dem_patches.shape[2],
                                                                dem_patches.shape[0] * dem_patches.shape[3],
                                                                dem_patches.shape[1] * dem_patches.shape[4]))
        #
        rec_patches = np.array(rec_patches)
        rec_patches = rec_patches.reshape((img.shape[1] // block_shape[0], img.shape[2] // block_shape[1],
                                           rec_patches.shape[1], rec_patches.shape[2], rec_patches.shape[3]))
        rec = np.moveaxis(rec_patches, [0, 1], [1, 3]).reshape((rec_patches.shape[2],
                                                                rec_patches.shape[0] * rec_patches.shape[3],
                                                                rec_patches.shape[1] * rec_patches.shape[4]))
        #
        if return_reconstruction:
            return dem, rec
        else:
            return dem

    def _get_model_path(self, model_name):
        model_path = os.path.join(Path.home(), '.deepem', model_name)
        os.makedirs(os.path.join(Path.home(), '.deepem'), exist_ok=True)
        if not os.path.exists(model_path):
            request.urlretrieve('http://kanzelhohe.uni-graz.at/iti/' + model_name, filename=model_path)
        return model_path
