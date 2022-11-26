import os

import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, AsinhStretch

from dem.train.callback import sdo_cmaps
from dem.train.generator import DEMDataset, FITSDEMDataset
from dem.train.model import DeepEM

evaluation_path = '/gpfs/gpfs0/robert.jarolim/dem/examples'
os.makedirs(evaluation_path, exist_ok=True)
dataset = FITSDEMDataset('/gpfs/gpfs0/robert.jarolim/data/dem_test_prep')

image = dataset[0]

norm = ImageNormalize(vmin=-1, vmax=1, stretch=AsinhStretch(0.005), clip=True)
for c_img, cmap in zip(image, sdo_cmaps):
    plt.imsave(f'{evaluation_path}/{cmap.name}.jpg', norm(c_img), cmap=cmap, origin='lower', vmin=0, vmax=1, )


dataset = DEMDataset('/gpfs/gpfs0/robert.jarolim/data/dem_test')

image = dataset[0]

norm = ImageNormalize(vmin=-1, vmax=1, stretch=AsinhStretch(0.005), clip=True)
for c_img, cmap in zip(image, sdo_cmaps):
    plt.imsave(f'{evaluation_path}/fd_{cmap.name}.jpg', norm(c_img), vmin=0, vmax=1, cmap=cmap, origin='lower')


base_path = '/gpfs/gpfs0/robert.jarolim/dem/version15'
dem_model = DeepEM(model_path=os.path.join(base_path, 'model.pt'))

dem, reconstruction = dem_model.compute(image, return_reconstruction=True)