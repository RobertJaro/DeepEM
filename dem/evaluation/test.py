import glob
import gzip
import os
import shutil
from copy import copy

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from sunpy.map import Map
from torch.utils.data import DataLoader

from dem.module import DEMModule
from dem.train.callback import sdo_cmaps
from dem.train.generator import FITSDEMDataset

base_path = '/gpfs/gpfs0/robert.jarolim/dem/version19'
data_dir = '/gpfs/gpfs0/robert.jarolim/data/dem_test_prep'
evaluation_path = os.path.join(base_path, 'evaluation')
fits_path = os.path.join(base_path, 'fits')

os.makedirs(evaluation_path, exist_ok=True)
os.makedirs(fits_path, exist_ok=True)

wls = ['94', '131', '171', '193', '211', '335']

dem_model = DEMModule(model_path=os.path.join(base_path, 'model.pt'))
T = dem_model.T

ds = FITSDEMDataset(data_dir, )

s_maps = [Map(f) for f in sorted(glob.glob('/gpfs/gpfs0/robert.jarolim/data/dem_test_prep/131/*.fits'))]

# load maps and data
loader = DataLoader(ds, batch_size=None, num_workers=os.cpu_count())
for idx, (input_image, ref_map) in enumerate(zip(loader, s_maps)):
    input_image = input_image.detach().numpy()
    # dem_model.fit_images([input_image], epochs=100)
    dem_result = dem_model.compute(input_image, uncertainty=True)
    image = input_image[:6]
    dem = dem_result['dem']
    dem_uncertainty = dem_result['dem_uncertainty'] if 'dem_uncertainty' in dem_result else np.zeros_like(dem)
    reconstruction = dem_result['reconstruction']
    #
    saturation_mask = np.max(image >= 5, axis=0)
    image[:, saturation_mask] = np.nan
    reconstruction[:, saturation_mask] = np.nan
    #
    image = (image + 1) / 2
    reconstruction = (reconstruction + 1) / 2
    print('Mean deviation [counts]:', np.nanmean(np.abs(reconstruction - image), (1, 2)) * 1e4)
    print('Mean deviation [%]:', np.nanmean(np.abs(reconstruction - image), (1, 2)) / np.nanmean(image, (1, 2)) * 100)
    print('Computing time:', dem_result['computing_time'])
    #
    plt.figure(figsize=(8, 4))
    plt.errorbar(T, (dem).mean((1, 2)), yerr=dem_uncertainty.mean((1, 2)), ecolor='red', capsize=2)
    plt.xlim(0, 25e6)
    plt.ylim(0, 2e22)
    plt.xlabel('T')
    plt.ylabel('DEM')
    plt.title('Average DEM')
    plt.savefig(os.path.join(evaluation_path, f'{idx}_dem.jpg'))
    plt.close()
    #
    fig, axs = plt.subplots(3, image.shape[0], figsize=(image.shape[0] * 3, 9))
    [ax.set_axis_off() for ax in np.ravel(axs)]
    for i, (ax, cmap) in enumerate(zip(axs[0], sdo_cmaps)):
        cmap = copy(cmap)
        cmap.set_bad('red')
        ax.imshow(image[i], cmap=cmap, vmin=0, vmax=1, origin='lower')
    #
    #
    # for i, ax in enumerate(axs[1]):
    #   ax.imshow(sum_dem[i], origin='lower', cmap='inferno')
    #
    #
    for i, (ax, cmap) in enumerate(zip(axs[1], sdo_cmaps)):
        cmap = copy(cmap)
        cmap.set_bad('red')
        ax.imshow(reconstruction[i], cmap=cmap, vmin=0, vmax=1, origin='lower')

    for i, ax in enumerate(axs[2]):
        ax.imshow((reconstruction[i] - image[i]) / np.mean(image[i]) * 100, cmap='bwr', vmin=-50, vmax=50,
                  origin='lower')
    #
    #
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_path, f'{idx}_reconstruction.jpg'))
    plt.close()
    # save dem
    t_mask = T <= 20e6
    for temp, dem_bin, dem_uc in zip(T[t_mask], dem[t_mask], dem_uncertainty[t_mask]):
        date_str = ref_map.date.to_datetime().isoformat('T')
        meta_info = {'DATE-OBS': date_str, 'TEMPERATURE': temp}
        # save fits
        fp = os.path.join(fits_path, '%s_TEMP%.02f.fits' % (date_str, temp))
        hdu = fits.PrimaryHDU(dem_bin)
        for i, v in meta_info.items():
            hdu.header[i] = v
        hdul = fits.HDUList([hdu])
        hdul.writeto(fp, overwrite=True)
        with open(fp, 'rb') as f_in, gzip.open(fp + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
        os.remove(fp)
        # save uncertainty
        fp = os.path.join(fits_path, '%s_TEMP%.02f_error.fits' % (date_str, temp))
        hdu = fits.PrimaryHDU(dem_uc)
        for i, v in meta_info.items():
            hdu.header[i] = v
        hdul = fits.HDUList([hdu])
        hdul.writeto(fp, overwrite=True)
        with open(fp, 'rb') as f_in, gzip.open(fp + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
        os.remove(fp)

shutil.make_archive(fits_path, 'zip', fits_path)
