import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from sunpy.visualization.colormaps import cm
from torch.utils.data import DataLoader

sdo_cmaps = [cm.sdoaia94, cm.sdoaia131, cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia335]


class PlotCallback():

    def __init__(self, data_set, model, prediction_path, temperatures, saturation_limit, device):
        self.data_set = data_set
        self.model = model
        self.prediction_path = prediction_path
        self.device = device
        self.temperatures = temperatures
        self.saturation_limit = saturation_limit

    def __call__(self, epoch):
        loader = DataLoader(self.data_set, batch_size=4, shuffle=False, num_workers=4)
        images = []
        reconstructions = []
        dems = []
        with torch.no_grad():
            for image in loader:
                reconstruction, dem = self.model(image.to(self.device))
                images.append(image[:, :6].detach().cpu().numpy())
                reconstructions.append(reconstruction.detach().cpu().numpy())
                dems.append(dem.detach().cpu().numpy())
        images = np.concatenate(images)
        reconstructions = np.concatenate(reconstructions)
        dems = np.concatenate(dems)
        n_rows = images.shape[0]
        n_dem_maps = 6
        n_columns = images.shape[1] + reconstructions.shape[1] + n_dem_maps + 1
        grid_ratios = np.ones((n_columns))
        grid_ratios[-1] = 3
        fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns, n_rows),
                                gridspec_kw={'width_ratios': grid_ratios})
        [ax.set_axis_off() for ax in np.ravel(axs)]
        for row, img, rec, dem in zip(axs, images, reconstructions, dems):
            for ax, c_img, cmap in zip(row[:img.shape[0]], img, sdo_cmaps):
                c_img[c_img >= self.saturation_limit] = np.nan
                ax.imshow(c_img, cmap=cmap, vmin=-1, vmax=1)
            row = row[img.shape[0]:]
            for ax, c_img, cmap in zip(row[:rec.shape[0]], rec, sdo_cmaps):
                ax.imshow(c_img, cmap=cmap, vmin=-1, vmax=1)
            row = row[rec.shape[0]:]
            sum_dem = block_reduce(dem, (dem.shape[0] // n_dem_maps, 1, 1))
            for ax, c_dem in zip(row[:n_dem_maps], sum_dem):
                ax.imshow(c_dem)
            row = row[n_dem_maps:]
            # row[0].plot(self.temperatures, (10 ** log_dem).sum(axis=(1, 2)))
            row[0].plot(self.temperatures, dem.mean(axis=(1, 2)))
            row[0].set_xlim(0, 20e6)
            row[0].set_ylim(0, 2e22)
            row[0].set_axis_on()
            row[0].tick_params(direction="in")
            row[0].axes.xaxis.set_ticklabels([])
        fig.tight_layout()
        fig.savefig(os.path.join(self.prediction_path, '%04d.jpg') % (epoch + 1), dpi=300)
        plt.close(fig)
