import os

import torch
from astropy.visualization import ImageNormalize, AsinhStretch
from sunpy.visualization.colormaps import cm
from torch.utils.data import DataLoader
import numpy as np

from matplotlib import pyplot as plt

sdo_cmaps = [cm.sdoaia94, cm.sdoaia131, cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia335]

class PlotCallback():

    def __init__(self, data_set, model, prediction_path, device):
        self.data_set = data_set
        self.model = model
        self.prediction_path = prediction_path
        self.device = device

    def __call__(self, epoch):
        loader = DataLoader(self.data_set, batch_size=4, shuffle=False, num_workers=8)
        images = []
        reconstructions = []
        dems = []
        with torch.no_grad():
            for image in loader:
                reconstruction, log_dem = self.model(image.to(self.device))
                images.append(image.detach().cpu().numpy())
                reconstructions.append(reconstruction.detach().cpu().numpy())
                dems.append(log_dem.detach().cpu().numpy())
        images = np.concatenate(images)
        reconstructions = np.concatenate(reconstructions)
        dems = np.concatenate(dems)
        n_rows = images.shape[0]
        n_dem_maps = 15
        n_columns = images.shape[1] + reconstructions.shape[1] + n_dem_maps
        fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns, n_rows))
        [ax.set_axis_off() for ax in np.ravel(axs)]
        for row, img, rec, dem in zip(axs, images, reconstructions, dems):
            for ax, c_img, cmap in zip(row[:img.shape[0]], img, sdo_cmaps):
                ax.imshow(c_img, cmap=cmap, norm=ImageNormalize(c_img, vmin=-1, vmax=1, stretch=AsinhStretch(0.005)))
            row = row[img.shape[0]:]
            for ax, c_img, cmap in zip(row[:rec.shape[0]], rec, sdo_cmaps):
                ax.imshow(c_img, cmap=cmap, norm=ImageNormalize(c_img, vmin=-1, vmax=1, stretch=AsinhStretch(0.005)))
            row = row[rec.shape[0]:]
            idx = np.linspace(0, len(dem) - 1, n_dem_maps).astype(int)
            for ax, c_dem in zip(row[:n_dem_maps], dem[idx]):
                ax.imshow(c_dem, vmin=26, vmax=29)
        fig.tight_layout()
        fig.savefig(os.path.join(self.prediction_path, '%04d.jpg') % (epoch + 1), dpi=80)
        plt.close(fig)

