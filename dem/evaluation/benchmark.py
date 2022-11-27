import glob
import os
import shutil
from datetime import datetime

import numpy as np
from astropy.visualization import ImageNormalize, AsinhStretch, LogStretch
from dateutil.parser import parse
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader

from dem.train.callback import sdo_cmaps
from dem.train.generator import AIADEMDataset
from dem.train.model import DeepEM

base_path = '/gpfs/gpfs0/robert.jarolim/dem/uc_version1'
data_dir = '/gpfs/gpfs0/robert.jarolim/data/dem_event'
evaluation_path = os.path.join(base_path, 'benchmark')
benchmarking_mode = True

os.makedirs(evaluation_path, exist_ok=True)

wls = ['94', '131', '171', '193', '211', '335']

dem_model = DeepEM(model_path=os.path.join(base_path, 'model.pt'))
logT = dem_model.log_T
T = 10 ** logT

ds = AIADEMDataset(data_dir, skip_register=True)

# load maps and data
loader = DataLoader(ds, batch_size=None, num_workers=4)
dates = [parse(os.path.basename(f)[:-5]).isoformat(' ') for f in sorted(glob.glob('/gpfs/gpfs0/robert.jarolim/data/dem_event/131/*.fits'))]

start_time = datetime.now()
for idx, (image, date_str) in enumerate(zip(loader, dates)):
    image = image.detach().numpy()
    dem_result = dem_model.compute(image, uncertainty=not benchmarking_mode)
    #
    if benchmarking_mode:
        print('Computing time:', dem_result['computing_time'])
        continue
    #
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
    fig = plt.figure(figsize=(8, 4), constrained_layout=True)
    gs = fig.add_gridspec(3, 6)

    for i, cmap in enumerate(sdo_cmaps):
        ax = fig.add_subplot(gs[0, i])
        ax.set_axis_off()
        ax.imshow(image[i], cmap=cmap, norm=ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch(0.005)), origin='lower')

    bins = 10 ** np.array([5.75, 6.05, 6.35, 6.65, 6.95, 7.25, 7.55])
    dem_integrals = [dem[(T > bins[i]) & (T < bins[i + 1])].sum(0) * (bins[i + 1] - bins[i])  for i in range(6)]
    axs = [fig.add_subplot(gs[1, i]) for i in range(6)]
    for i in range(6):
        ax = axs[i]
        ax.set_axis_off()
        pc = ax.imshow(dem_integrals[i], cmap='viridis', norm=ImageNormalize(vmin=0, vmax=5e29, stretch=LogStretch()), origin='lower')
        ax.set_title('%.01f - %.01f MK' % (bins[i] * 1e-6, bins[i + 1] * 1e-6), fontsize='medium')
    #
    # cbar = fig.colorbar(pc, ax=axs)
    # cbar.ax.set_ylabel('EM [1e29 cm$^{-5}$]', rotation=270)

    ax = fig.add_subplot(gs[2, :])
    ax.set_title(date_str)
    ax.errorbar(T * 1e-6, dem.sum((1, 2)), yerr=dem_uncertainty.sum((1, 2)), ecolor='red', capsize=2)
    ax.set_xlim(0, 25)
    ax.set_ylim(1e24, 5e26)
    ax.set_xlabel('T [MK]')
    ax.set_ylabel('DEM [cm$^{-5}$ K$^{-1}$]')
    ax.semilogy()
    fig.savefig(os.path.join(evaluation_path, f'%03d.jpg' % idx), dpi=300)
    plt.close(fig)

print('Total computing time:', datetime.now() - start_time)
shutil.make_archive(evaluation_path, 'zip', evaluation_path)