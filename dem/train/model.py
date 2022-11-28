import os
from datetime import datetime
from pathlib import Path
from urllib import request

import numpy as np
import torch
from astropy.nddata import block_reduce
from skimage.util import view_as_blocks
from torch import nn
from torch.utils.data import DataLoader


class DEMModel(nn.Module):

    def __init__(self, channels, n_normal, log_T, k, normalization, n_dims = 512, scaling_factor=1e28):
        super().__init__()
        #
        self.register_buffer("k", k)
        self.register_buffer("scaling_factor", torch.tensor(scaling_factor, dtype=torch.float32))
        self.register_buffer("normalization", normalization)

        self.register_buffer("logT", torch.tensor(log_T, dtype=torch.float32))
        #
        T = 10 ** log_T
        self.register_buffer("dT", torch.tensor(np.gradient(T), dtype=torch.float32))
        #
        convs = []
        convs += [nn.Conv2d(channels, n_dims, 3, padding=1, padding_mode='reflect'), Sine(), ]
        for _ in range(4):
            convs += [nn.Conv2d(n_dims, n_dims, 3, padding=1, padding_mode='reflect'), Sine(), nn.Dropout2d(0.25), ]
        self.convs = nn.Sequential(*convs)
        self.out = nn.Conv2d(n_dims, n_normal * 3, 3, padding=1, padding_mode='reflect')
        #
        self.out_act = nn.Softplus()

    def forward(self, x, logT=None):
        logT = logT if logT is not None else self.logT  # default log T
        d_logT = logT[1] - logT[0]  # assumes equally spaced temperature bins
        #
        em = self._compute_dem(x, logT) * d_logT
        euv_normalized = self.compute_euv(em)
        #
        dem = em / self.dT[:, None, None] * self.scaling_factor  # compute DEM from EM (use correct T bins)
        #
        return euv_normalized, dem

    def compute_euv(self, em):
        y = torch.einsum('ijkl,jm->imkl', em, self.k * self.scaling_factor)  # 1e28 DN / s / px
        euv_normalized = y / self.normalization[None, :, None, None]  # scale to [0, 1]
        euv_normalized = (euv_normalized * 2) - 1  # scale approx. [-1, 1]
        # echte DEM = EM / [K bin] = dem * dlogT / [K bin]
        return euv_normalized

    def _compute_dem(self, x, logT):
        # transform to DEM
        x = self.convs(x)
        x = self.out(x)
        x = x.view(x.shape[0], -1, 3, *x.shape[2:])
        # (batch, n_normal, T_bins, w, h)
        # min width of 1 temperature bin
        std = self.out_act(x[:, :, 0, None, :, :]) + 0.1
        mean = torch.sigmoid(x[:, :, 1, None, :, :]) * (self.logT.max() - self.logT.min()) + self.logT.min()
        w = self.out_act(x[:, :, 2, None, :, :])
        logT = logT[None, None, :, None, None]  # (batch, n_normal, T_bins, w, h)
        normal = w * (std * np.sqrt(2 * np.pi) + 1e-8) ** -1 * torch.exp(-0.5 * (logT - mean) ** 2 / (std ** 2 + 1e-8))
        dem = normal.sum(1)
        return dem


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class DEM:

    def __init__(self, model_path=None, model_name='dem_v0_1.pt', device=None):
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu') if not device else device
        self.model = torch.load(model_path, map_location=device) if model_path else \
            torch.load(self._get_model_path(model_name), map_location=device)
        self.device = device
        self.log_T = self.model.logT.cpu().numpy()

    def compute(self, image, log_T=None, bin=None, uncertainty=False, n_uncertainty_ensemble=20):
        with torch.no_grad():
            start_time = datetime.now()
            image = torch.tensor(image, dtype=torch.float32).to(self.device)
            image = image[None,]  # expand batch dimension
            log_T = torch.tensor(log_T, dtype=torch.float32).to(self.device) if log_T is not None else None
            if uncertainty:
                self.model.train()
                reconstruction, dem = [], []
                for _ in range(n_uncertainty_ensemble):
                    r, d = self.model(image, log_T)
                    reconstruction += [r.cpu()]
                    dem += [d.cpu()]
                reconstruction, dem = torch.cat(reconstruction), torch.cat(dem)
                # reconstruction uncertainty
                reconstruction_uncertainty = torch.std(reconstruction, dim=0).cpu().numpy()
                # dem + uncertainty
                dem_uncertainty = torch.std(dem, dim=0).cpu().numpy()
                dem = torch.mean(dem, dim=0).cpu().numpy()
                # calculate reference reconstruction
                self.model.eval()
                reconstruction, _ = self.model(image, log_T)
                reconstruction = reconstruction.cpu().numpy()[0]
            else:
                self.model.eval()
                reconstruction, dem = self.model(image, log_T)
                reconstruction = reconstruction.cpu().numpy()[0]
                dem = dem.cpu().numpy()[0]
            if bin:
                dem = block_reduce(dem, (1, bin, bin), func=np.mean)
                reconstruction = block_reduce(reconstruction, (1, bin, bin), func=np.mean)
                if uncertainty:
                    dem_uncertainty = block_reduce(dem_uncertainty, (1, bin, bin), func=np.mean)
                    reconstruction_uncertainty = block_reduce(reconstruction_uncertainty, (1, bin, bin), func=np.mean)
            # build result
            result = {'dem': dem, 'reconstruction': reconstruction, 'computing_time': datetime.now() - start_time}
            if uncertainty:
                result.update(
                    {'dem_uncertainty': dem_uncertainty, 'reconstruction_uncertainty': reconstruction_uncertainty})
            return result

    def icompute(self, dataset, num_workers=None, block_shape=(512, 512), **kwargs):
        num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
        #
        loader = DataLoader(dataset, batch_size=1, num_workers=num_workers)
        for image in loader:
            if block_shape:
                yield self.compute_patches(image[0].numpy(), block_shape, **kwargs)
            else:
                yield self.compute(image[0], **kwargs)

    def fit(self, images):
        raise NotImplementedError('Few-shot learning is not implemented.')

    def compute_patches(self, img, block_shape, **kwargs):
        start_time = datetime.now()
        patch_shape = (img.shape[0], *block_shape)
        patches = view_as_blocks(img, patch_shape)
        patches = np.reshape(patches, (-1, *patch_shape))
        results = []
        with torch.no_grad():
            for patch in patches:
                r = self.compute(patch, **kwargs)
                print(r['computing_time'])
                del r['computing_time']
                results += [r]
        #
        d = {}
        for k in results[0].keys():
            patches = [d[k] for d in results]
            patches = np.array(patches)
            patches = patches.reshape((img.shape[1] // block_shape[0], img.shape[2] // block_shape[1],
                                       patches.shape[1], patches.shape[2], patches.shape[3]))
            r = np.moveaxis(patches, [0, 1], [1, 3]).reshape((patches.shape[2],
                                                              patches.shape[0] * patches.shape[3],
                                                              patches.shape[1] * patches.shape[4]))
            d[k] = r
        d['computing_time'] = datetime.now() - start_time
        #
        return d

    def _get_model_path(self, model_name):
        model_path = os.path.join(Path.home(), '.deepem', model_name)
        os.makedirs(os.path.join(Path.home(), '.deepem'), exist_ok=True)
        if not os.path.exists(model_path):
            request.urlretrieve('http://kanzelhohe.uni-graz.at/iti/' + model_name, filename=model_path)
        return model_path
