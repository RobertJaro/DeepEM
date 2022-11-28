import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.visualization import ImageNormalize, AsinhStretch

from dem.train.callback import sdo_cmaps
from dem.train.generator import FITSDEMDataset
from dem.train.model import DEM

evaluation_path = '/gpfs/gpfs0/robert.jarolim/dem/examples'
os.makedirs(evaluation_path, exist_ok=True)
dataset = FITSDEMDataset('/gpfs/gpfs0/robert.jarolim/data/dem_test_prep')

image = dataset[0]

norm = ImageNormalize(vmin=-1, vmax=1, stretch=AsinhStretch(0.005), clip=True)
for c_img, cmap in zip(image, sdo_cmaps):
    plt.imsave(f'{evaluation_path}/{cmap.name}.jpg', norm(c_img), cmap=cmap, origin='lower', vmin=0, vmax=1, )

base_path = '/gpfs/gpfs0/robert.jarolim/dem/uc_version1'
dem_model = DEM(model_path=os.path.join(base_path, 'model.pt'))

result = dem_model.compute(image, uncertainty=True)
dem = result['dem']
dem_uncertainty = result['dem_uncertainty']
reconstruction = result['reconstruction']

vmax = np.abs(dem).max()
for i, c_img in enumerate(dem):
    plt.imsave(f'{evaluation_path}/dem_%03d.jpg' % i, c_img, cmap='viridis', origin='lower', vmin=0, vmax=vmax)

plt.imsave(f'{evaluation_path}/dem_sum.jpg', np.log10(dem.sum(0)), cmap='viridis', origin='lower')

norm = ImageNormalize(vmin=-1, vmax=1, stretch=AsinhStretch(0.005), clip=True)
for c_img, cmap in zip(reconstruction, sdo_cmaps):
    plt.imsave(f'{evaluation_path}/reconstruction_{cmap.name}.jpg', norm(c_img), cmap=cmap, origin='lower', vmin=0, vmax=1, )

logT = dem_model.log_T

plt.figure(figsize=(6, 3))
plt.errorbar(10 ** logT * 1e-6, dem.sum((1, 2)), yerr=dem_uncertainty.sum((1, 2)), ecolor='red', capsize=2)
plt.xlim(0, 25)
plt.ylim(0, None)
plt.xlabel('T [MK]')
plt.ylabel('DEM')
plt.title('Total DEM')
plt.tight_layout()
plt.savefig(os.path.join(evaluation_path, f'dem.jpg'), dpi=300)
plt.close()

temperature_response = pd.read_csv('/gpfs/gpfs0/robert.jarolim/data/dem/aia_temperature_response_2013.csv').to_numpy()
# select smaller temperature range
t_filter = (temperature_response[:, 0] >= 5.7) & (temperature_response[:, 0] <= 7.4)#(temperature_response[:, 0] >= 5.5) & (temperature_response[:, 0] <= 7.4)
temperature_response = temperature_response[t_filter]

channel_response = temperature_response[:, 1:]
channel_response = np.delete(channel_response, 5, 1)  # remove 304
channel_response[channel_response < 0] = 0 # adjust invalid values

logT = temperature_response[:, 0]

legends = ['94', '131', '171', '193', '211', '335']
colors = ['green', 'cyan', 'yellow', 'gold', 'violet', 'blue']
plt.figure(figsize=(6, 3))
for i in range(6):
    plt.plot(10 ** logT, channel_response[:, i], label=legends[i], color=colors[i])


plt.xlabel('Temperature [K]')
plt.ylabel('$K_{ij}$ [DN cm$^{-5}$ pix$^{-1}$]')
plt.ylim(1e-28)
plt.title('Temperature Response')
plt.loglog()
plt.legend(loc='upper right', fancybox=True, shadow=True, borderpad=1)
plt.tight_layout()
plt.savefig(os.path.join(evaluation_path, f'temperature_response.png'), dpi=300, transparent=True)
plt.close()