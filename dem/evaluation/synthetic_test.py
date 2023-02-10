import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io.fits import getdata

from dem.train.generator import sdo_norms
from dem.train.model import DEM

base_path = '/gpfs/gpfs0/robert.jarolim/dem/uc_version9_log_tanh_shift_z001'
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

dems, dem_uncertainties = [], []
for fits_files in [fits_files_1, fits_files_2, fits_files_3, fits_files_4]:
    dem_model = DEM(model_path=os.path.join(base_path, 'model.pt'))
    logT = dem_model.log_T

    image = np.array([norm(getdata(f)) * 2 - 1 for f, norm in zip(fits_files, sdo_norms.values())])
    dem_model.fit(image)
    result = dem_model.compute(image, uncertainty=True)
    dems += [result['dem']]
    dem_uncertainties += [result['dem_uncertainty']]

plt.figure(figsize=(4, 4))
for i, (dem, dem_uncertainty) in enumerate(zip(dems, dem_uncertainties)):
    plt.errorbar(10 ** logT * 1e-6, dem.mean((1, 2)), yerr=dem_uncertainty.mean((1, 2)), ecolor='red', capsize=2,
                 label=f'Test Case {i + 1}')
plt.xlim(0, 20)
plt.ylim(0, 2e22)
plt.xlabel('T [MK]')
plt.ylabel('DEM')
plt.title('Total DEM')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(evaluation_path, f'dem.jpg'), dpi=300)
plt.close()
