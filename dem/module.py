import os
from datetime import datetime
from pathlib import Path
from urllib import request

import numpy as np
import torch
from astropy.nddata import block_reduce
from skimage.util import view_as_blocks
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


class DEMModule:

    def __init__(self, model_path=None, model=None, model_name='dem_v0_1.pt', device=None, lambda_zo=1e-4):
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu') if not device else device
        if model:  # create plain
            model = model
        else:
            model_path = model_path if model_path else torch.load(self._get_model_path(model_name), map_location=device)
            model = torch.load(model_path, map_location=device)

        self.parallel_model = nn.DataParallel(model)
        self.device = device
        self.logT = model.logT.cpu().numpy()
        self.lambda_zo = lambda_zo
        self.saturation_limit = 5

    def compute(self, image, log_T=None, bin=None, uncertainty=False, n_uncertainty_ensemble=20):
        with torch.no_grad():
            start_time = datetime.now()
            image = torch.tensor(image, dtype=torch.float32).to(self.device)
            image = image[None,]  # expand batch dimension
            log_T = torch.tensor(log_T, dtype=torch.float32).to(self.device) if log_T is not None else None
            if uncertainty:
                self.parallel_model.train()
                reconstruction, dem = [], []
                for _ in range(n_uncertainty_ensemble):
                    r, d = self.parallel_model(image, log_T)
                    reconstruction += [r.cpu()]
                    dem += [d.cpu()]
                reconstruction, dem = torch.cat(reconstruction), torch.cat(dem)
                # reconstruction uncertainty
                reconstruction_uncertainty = torch.std(reconstruction, dim=0).cpu().numpy()
                reconstruction = torch.mean(reconstruction, dim=0).cpu().numpy()
                # dem + uncertainty
                dem_uncertainty = torch.std(dem, dim=0).cpu().numpy()
                dem = torch.mean(dem, dim=0).cpu().numpy()
                # calculate reference reconstruction
                # self.parallel_model.eval()
                # reconstruction, dem = self.parallel_model(image, log_T)
                # reconstruction = reconstruction.cpu().numpy()[0]
                # dem = dem.cpu().numpy()[0]
            else:
                self.parallel_model.eval()
                reconstruction, dem = self.parallel_model(image, log_T)
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

    def fit_image(self, image, block_shape=None, batch_size=1, epochs=int(1e3)):
        #
        if block_shape:
            patch_shape = (image.shape[0], *block_shape)
            patches = view_as_blocks(image, patch_shape)
            patches = np.reshape(patches, (-1, *patch_shape))
            train_data = [patches[idx * batch_size: (idx + 1) * batch_size]
                          for idx in range(np.ceil(len(patches) / batch_size).astype(np.int))]
        else:
            train_data = [image]
        train_data = torch.tensor(train_data, dtype=torch.float32)

        optimizer = Adam(self.parallel_model.parameters(), lr=1e-4)

        self.parallel_model.train()
        for epoch in range(epochs):
            for idx in range(np.ceil(len(train_data) / batch_size).astype(np.int)):
                optimizer.zero_grad()
                batch = train_data[idx * batch_size: (idx + 1) * batch_size].to(self.device)
                # forward
                reconstructed_image, dem = self.parallel_model(batch)
                # compute loss
                reconstruction_loss, zeroth_order_regularization = self._compute_loss(dem, reconstructed_image, batch)
                # backward
                total_loss = reconstruction_loss + zeroth_order_regularization * self.lambda_zo
                assert not torch.isnan(total_loss)
                total_loss.backward()
                reconstruction_loss.detach().cpu().numpy(), zeroth_order_regularization.detach().cpu().numpy()
                optimizer.step()

    def _compute_loss(self, dem, reconstructed_image, ref_image):
        lf = nn.HuberLoss(reduction='none')
        # channel_weight = torch.tensor([100, 10, 1, 1, 1, 20], dtype=torch.float32, device=dem.device)[None, :, None, None]
        saturation_mask = torch.min(ref_image < self.saturation_limit, dim=1, keepdim=True)[0]
        reconstruction_loss = (lf(reconstructed_image, ref_image) * saturation_mask).mean()
        zeroth_order_regularization = torch.mean((dem / 1e23) * saturation_mask)
        return reconstruction_loss, zeroth_order_regularization

    def _get_model_path(self, model_name):
        model_path = os.path.join(Path.home(), '.deepem', model_name)
        os.makedirs(os.path.join(Path.home(), '.deepem'), exist_ok=True)
        if not os.path.exists(model_path):
            request.urlretrieve('http://kanzelhohe.uni-graz.at/iti/' + model_name, filename=model_path)
        return model_path