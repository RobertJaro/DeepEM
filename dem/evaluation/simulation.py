import gzip
import os
import shutil
from copy import copy

import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch, LogStretch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dem.train.callback import sdo_cmaps
from dem.train.generator import FITSDEMDataset
from dem.train.model import DeepEM

base_path = '/gpfs/gpfs0/robert.jarolim/dem/uc_version1'
evaluation_path = os.path.join(base_path, 'evaluation_sim')
fits_path = os.path.join(base_path, 'fits_sim')

os.makedirs(evaluation_path, exist_ok=True)
os.makedirs(fits_path, exist_ok=True)

dem_model = DeepEM(model_path=os.path.join(base_path, 'model.pt'))
logT = dem_model.log_T
T = 10 ** logT

ds = FITSDEMDataset([['/gpfs/gpfs0/robert.jarolim/data/dem/simulation_prep/94A_Evenorm.fits'],
                     ['/gpfs/gpfs0/robert.jarolim/data/dem/simulation_prep/131A_Evenorm.fits'],
                     ['/gpfs/gpfs0/robert.jarolim/data/dem/simulation_prep/171A_Evenorm.fits'],
                     ['/gpfs/gpfs0/robert.jarolim/data/dem/simulation_prep/193A_Evenorm.fits'],
                     ['/gpfs/gpfs0/robert.jarolim/data/dem/simulation_prep/211A_Evenorm.fits'],
                     ['/gpfs/gpfs0/robert.jarolim/data/dem/simulation_prep/335A_Evenorm.fits'], ])

# load maps and data
loader = DataLoader(ds, batch_size=None, num_workers=os.cpu_count())
for idx, image in enumerate(loader):
    image = image.detach().numpy()
    dem_result = dem_model.compute(image, uncertainty=True)
    dem, reconstruction = dem_result['dem'], dem_result['reconstruction']
    dem_uncertainty = dem_result['dem_uncertainty'] if 'dem_uncertainty' in dem_result else np.zeros_like(dem)
    #
    saturation_mask = np.max(image >= 10, axis=0)
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
    plt.errorbar(10 ** logT, (dem).sum((1, 2)), yerr=dem_uncertainty.sum((1, 2)), ecolor='red', capsize=2)
    plt.xlim(0, 25e6)
    plt.ylim(0, 5e26)
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
        ax.imshow(image[i], cmap=cmap, norm=ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch(0.005)), origin='lower')
    #
    #
    bins = 10 ** np.array([5.75, 6.05, 6.35, 6.65, 6.95, 7.25, 7.55])
    dem_integrals = [dem[(T > bins[i]) & (T < bins[i + 1])].sum(0) * (bins[i + 1] - bins[i])  for i in range(6)]
    for i, ax in enumerate(axs[1]):
        ax.imshow(dem_integrals[i], cmap='viridis', norm=ImageNormalize(vmin=0, vmax=5e29, stretch=LogStretch()), origin='lower')
    #
    #
    for i, (ax, cmap) in enumerate(zip(axs[2], sdo_cmaps)):
        cmap = copy(cmap)
        cmap.set_bad('red')
        ax.imshow(reconstruction[i], cmap=cmap, norm=ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch(0.005)),
                  origin='lower')
    #
    #
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_path, f'{idx}_reconstruction.jpg'))
    plt.close()
    # save dem
    for temp, dem_bin, dem_unc in zip(logT, dem, dem_uncertainty):
        date_str = "2009-02-13"
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
        # save fits
        fp = os.path.join(fits_path, '%s_TEMP%.02f_error.fits' % (date_str, temp))
        hdu = fits.PrimaryHDU(dem_unc)
        for i, v in meta_info.items():
            hdu.header[i] = v
        hdul = fits.HDUList([hdu])
        hdul.writeto(fp, overwrite=True)
        with open(fp, 'rb') as f_in, gzip.open(fp + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
        os.remove(fp)

shutil.make_archive(fits_path, 'zip', fits_path)
