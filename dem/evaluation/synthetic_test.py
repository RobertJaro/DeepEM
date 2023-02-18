import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.io.fits import getdata

from dem.train.generator import sdo_norms
from dem.train.model import DEMModel
from dem.module import DEMModule

base_path = '/gpfs/gpfs0/robert.jarolim/dem/version4'
model_path = os.path.join(base_path, 'model.pt')
evaluation_path = os.path.join(base_path, 'synthetic_test')
os.makedirs(evaluation_path, exist_ok=True)

fits_files_1 = [
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/1/94A_Position1logT_6.10000_Position2logT_6.60000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/1/131A_Position1logT_6.10000_Position2logT_6.60000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/1/171A_Position1logT_6.10000_Position2logT_6.60000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/1/193A_Position1logT_6.10000_Position2logT_6.60000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/1/211A_Position1logT_6.10000_Position2logT_6.60000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/1/335A_Position1logT_6.10000_Position2logT_6.60000_FitsFromSyntheticDEM.fits',
]

fits_files_2 = [
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/2/94A_Position1logT_6.10000_Position2logT_6.80000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/2/131A_Position1logT_6.10000_Position2logT_6.80000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/2/171A_Position1logT_6.10000_Position2logT_6.80000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/2/193A_Position1logT_6.10000_Position2logT_6.80000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/2/211A_Position1logT_6.10000_Position2logT_6.80000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/2/335A_Position1logT_6.10000_Position2logT_6.80000_FitsFromSyntheticDEM.fits',
]

fits_files_3 = [
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/3/94A_Position1logT_6.10000_Position2logT_7.00000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/3/131A_Position1logT_6.10000_Position2logT_7.00000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/3/171A_Position1logT_6.10000_Position2logT_7.00000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/3/193A_Position1logT_6.10000_Position2logT_7.00000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/3/211A_Position1logT_6.10000_Position2logT_7.00000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/3/335A_Position1logT_6.10000_Position2logT_7.00000_FitsFromSyntheticDEM.fits',
]

fits_files_4 = [
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/4/94A_Position1logT_6.10000_Position2logT_7.20000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/4/131A_Position1logT_6.10000_Position2logT_7.20000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/4/171A_Position1logT_6.10000_Position2logT_7.20000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/4/193A_Position1logT_6.10000_Position2logT_7.20000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/4/211A_Position1logT_6.10000_Position2logT_7.20000_FitsFromSyntheticDEM.fits',
    '/gpfs/gpfs0/robert.jarolim/data/dem/SyntheticTestCases/4/335A_Position1logT_6.10000_Position2logT_7.20000_FitsFromSyntheticDEM.fits',
]

# temperature_response_path = '/gpfs/gpfs0/robert.jarolim/data/dem/aia_temperature_response_2013.csv'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# temperature_response = pd.read_csv(temperature_response_path).to_numpy()
#
# t_filter = (temperature_response[:, 0] >= 5.7) & (temperature_response[:, 0] <= 7.4)
# temperature_response = temperature_response[t_filter]
# channel_response = temperature_response[:, 1:]
# channel_response = np.delete(channel_response, 5, 1)  # remove 304
#
# temperatures_original = temperature_response[:, 0]
# temperatures = torch.arange(temperatures_original.min(), temperatures_original.max(), 0.01)
# channel_response = [np.interp(temperatures, temperatures_original, channel_response[:, i])
#                     for i in range(channel_response.shape[1])]
# channel_response = np.stack(channel_response, 1)
#
# k = torch.tensor(channel_response, dtype=torch.float32).to(device)
# k[k < 0] = 0  # adjust invalid values
# logT = torch.tensor(temperatures, dtype=torch.float32).to(device)
#
# normalization = torch.from_numpy(np.array([norm.vmax for norm in sdo_norms.values()])).float()
#
# zero_weighting = -torch.log10(k.sum(1)).to(device)
# zero_weighting = (zero_weighting - zero_weighting.min()) / (zero_weighting.max() - zero_weighting.min())
#
# model = DEMModel(channel_response.shape[1], 3, logT, k, normalization, n_dims=128)

dem_model = DEMModule(model_path=model_path)
logT = dem_model.logT

dems, dem_uncertainties = [], []
for fits_files in [fits_files_1, fits_files_2, fits_files_3, fits_files_4]:
    #
    image = np.array([norm(getdata(f)) * 2 - 1 for f, norm in zip(fits_files, sdo_norms.values())])
    # dem_model.fit_image(image)
    result = dem_model.compute(image, uncertainty=True)
    dems += [result['dem']]
    dem_uncertainties += [result['dem_uncertainty']]

    image = (image + 1) / 2
    result['reconstruction'] = (result['reconstruction'] + 1) / 2
    print((np.abs(result['reconstruction'] - image).mean((1, 2)) / np.mean(image, (1, 2))) * 100)
    print(np.abs(result['reconstruction'] - image).mean((1, 2)))

plt.figure(figsize=(4, 4))
for i, (dem, dem_uncertainty) in enumerate(zip(dems, dem_uncertainties)):
    plt.errorbar(10 ** logT * 1e-6, dem.mean((1, 2)), yerr=dem_uncertainty.mean((1, 2)), ecolor='red', capsize=2,
                 label=f'Test Case {i + 1}')

plt.xlim(0, 20)
plt.ylim(0, 2e22)
plt.axvline(10 ** .1, linestyle='--', color='black')
plt.axvline(10 ** .6, linestyle='--', color='tab:blue')
plt.axvline(10 ** .8, linestyle='--', color='tab:orange')
plt.axvline(10 ** 1., linestyle='--', color='tab:green')
plt.axvline(10 ** 1.2, linestyle='--', color='tab:red')
plt.xlabel('T [MK]')
plt.ylabel('DEM')
plt.title('Total DEM')
plt.legend()
plt.tight_layout()
# plt.loglog()
plt.savefig(os.path.join(evaluation_path, f'dem.jpg'), dpi=300)
plt.close()
