import glob
import os
import shutil
from datetime import datetime

import numpy as np
from astropy.visualization import ImageNormalize, AsinhStretch, LogStretch
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader

from dem.train.callback import sdo_cmaps
from dem.train.generator import AIADEMDataset
from dem.module import DEMModule

base_path = '/gpfs/gpfs0/robert.jarolim/dem/uc_version4'
data_dir = '/gpfs/gpfs0/robert.jarolim/data/dem_test'
evaluation_path = os.path.join(base_path, 'full_disk')
benchmarking_mode = True

os.makedirs(evaluation_path, exist_ok=True)

wls = ['94', '131', '171', '193', '211', '335']

dem_model = DEMModule(model_path=os.path.join(base_path, 'model.pt'))
logT = dem_model.log_T
T = 10 ** logT

ds = AIADEMDataset(data_dir, skip_register=True)

# load maps and data
image = ds[0]
dem_result = dem_model.compute_patches(image, (512, 512), uncertainty=False, bin=1)

dem = dem_result['dem']
image = (image + 1) / 2

print('Total computing time:', dem_result['computing_time'])

for i, (cmap, title) in enumerate(zip(sdo_cmaps, [94, 131, 171, 193, 211, 335])):
    norm = ImageNormalize(vmin=0, stretch=AsinhStretch(0.005))
    plt.imsave(os.path.join(evaluation_path, f'euv_{title}.jpg') , norm(image[i]), cmap=cmap, vmin=0, vmax=1, origin='lower')

bins = 10 ** np.array([5.75, 6.05, 6.35, 6.65, 6.95, 7.25, 7.55])
dem_integrals = [dem[(T > bins[i]) & (T < bins[i + 1])].sum(0) * (bins[i + 1] - bins[i]) for i in range(6)]
for i in range(6):
    fp = os.path.join(evaluation_path, 'tbin_%.01f_%.01f.jpg' % (bins[i] * 1e-6, bins[i + 1] * 1e-6))
    norm = ImageNormalize(vmin=0, vmax=5e29, stretch=LogStretch())
    plt.imsave(fp, norm(dem_integrals[i]), vmin=0, vmax=1, cmap='viridis', origin='lower')

mean_T = (dem * T[:, None, None] * np.gradient(T[:, None, None], axis=0)).sum(0) / ((dem * np.gradient(T[:, None, None], axis=0)).sum(0) + 1e-6)

fig, ax = plt.subplots(1, 1, figsize=(16, 16))
im = ax.imshow(mean_T, cmap='plasma', norm=LogNorm(vmin=10 ** 6.2, vmax = 10 ** 7.4), origin='lower')
ax.set_axis_off()
divider = make_axes_locatable(ax)
# Add an Axes to the right of the main Axes.
cax = divider.append_axes("right", size="7%", pad="2%")
cb = fig.colorbar(im, cax=cax)
cb.ax.set_ylabel('Temperature [K]', rotation=270, labelpad=30, fontsize=28)
ticks = [2e6, 6e6, 1e7, 2e7]
cb.set_ticks(ticks)
cb.set_ticklabels(['$2\cdot10^6$', '$8\cdot10^6$', '$1\cdot10^7$', '$2\cdot10^7$'])
cb.ax.tick_params(labelsize=28)
fig.tight_layout()
fig.savefig(os.path.join(evaluation_path, 'temperature.jpg'), dpi=300)
plt.close(fig)