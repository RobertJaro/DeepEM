import glob
import os
from copy import copy

import numpy as np

from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from sunpy.map import Map

from dem.train.callback import sdo_cmaps
from dem.train.generator import prep_aia_map, AIADEMDataset
from dem.module import DEMModule

base_path = '/gpfs/gpfs0/robert.jarolim/dem/version12'
data_dir = '/gpfs/gpfs0/robert.jarolim/data/dem'
evaluation_path = os.path.join(base_path, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

wls = ['94', '131', '171', '193', '211', '335']

dem_model = DEMModule(model_path=os.path.join(base_path, 'model.pt'))

# crop and dem

bl = u.Quantity((2000, 2000) * u.pix)
tr = u.Quantity((2500, 2500) * u.pix)
# select the first observation
files = [sorted(glob.glob(os.path.join(data_dir, wl, '*.fits')))[400] for wl in wls]

# load maps and data
maps = [prep_aia_map(Map(f).submap(bottom_left=bl, top_right=tr)) for f in files]
image = np.array([s_map.data for s_map in maps])

dem, reconstruction = dem_model.compute(image, return_reconstruction=True)

saturation_mask = np.max(image >= 5, axis=0)
image[:, saturation_mask] = np.nan
reconstruction[:, saturation_mask] = np.nan

image = (image + 1) / 2
reconstruction = (reconstruction + 1) / 2
print('Mean deviation [counts]:', np.nanmean(np.abs(reconstruction - image), (1, 2)) * 1e4)
print('Mean deviation [%]:', np.nanmean(np.abs(reconstruction - image), (1, 2)) / np.nanmean(image, (1, 2)) * 100)


plt.figure(figsize=(8, 4))
plt.plot(10 ** np.arange(4, 9.01, 0.05), (dem).sum((1, 2)))
plt.xlim(0, 25e6)
# plt.semilogy()
plt.xlabel('T')
plt.ylabel('DEM')
plt.title('Average DEM')
plt.savefig(os.path.join(evaluation_path, 'dem.jpg'))
plt.close()

# sum temperature bins
sum_dem = block_reduce(dem, (10, 1, 1))

fig, axs = plt.subplots(3, image.shape[0], figsize=(image.shape[0] * 3, 9))
[ax.set_axis_off() for ax in np.ravel(axs)]
for i, (ax, cmap) in enumerate(zip(axs[0], sdo_cmaps)):
  cmap = copy(cmap)
  cmap.set_bad('red')
  ax.imshow(image[i], cmap=cmap, norm=ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch(0.005)), origin='lower')


for i, ax in enumerate(axs[1]):
  ax.imshow(sum_dem[i], origin='lower', cmap='inferno')


for i, (ax, cmap) in enumerate(zip(axs[2], sdo_cmaps)):
  cmap = copy(cmap)
  cmap.set_bad('red')
  ax.imshow(reconstruction[i], cmap=cmap, norm=ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch(0.005)), origin='lower')


plt.tight_layout()
plt.savefig(os.path.join(evaluation_path, 'reconstruction.jpg'))
plt.close()