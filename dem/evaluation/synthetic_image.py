import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from aiapy.calibrate import estimate_error
from astropy import units as u
from scipy.io import readsav
from sklearn.utils import shuffle
from tqdm import tqdm

from dem.module import DEMModule
from dem.train.generator import sdo_norms
from dem.train.model import DEMModel

base_path = '/gpfs/gpfs0/robert.jarolim/dem/version12'
model_path = os.path.join(base_path, 'model.pt')
evaluation_path = os.path.join(base_path, 'synthetic_test')
os.makedirs(evaluation_path, exist_ok=True)

temperature_response_path = '/gpfs/gpfs0/robert.jarolim/data/dem/aia_temperature_response_2013.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
temperature_response = pd.read_csv(temperature_response_path).to_numpy()

t_filter = (temperature_response[:, 0] >= 5.7) & (temperature_response[:, 0] <= 7.4)
temperature_response = temperature_response[t_filter]
channel_response = temperature_response[:, 1:]
channel_response = np.delete(channel_response, 5, 1)  # remove 304

temperatures_original = temperature_response[:, 0]
temperatures = np.arange(10 ** temperatures_original.min(), 10 ** temperatures_original.max(), 1e4)
channel_response = [np.interp(temperatures, 10 ** temperatures_original, channel_response[:, i])
                    for i in range(channel_response.shape[1])]
channel_response = np.stack(channel_response, 1)

k = torch.tensor(channel_response, dtype=torch.float32).to(device)
k[k < 0] = 0  # adjust invalid values
T = torch.tensor(np.array(temperatures), dtype=torch.float32).to(device)

normalization = torch.from_numpy(np.array([norm.vmax for norm in sdo_norms.values()])).float()

dT = 1e4
means, stds = np.meshgrid(np.linspace(temperatures.min(), temperatures.max(), 100), np.linspace(1e4, 1e6, 100))
dems = 1e22 * (np.sqrt(2 * np.pi) * stds[None]) ** -1 * np.exp(-0.5 * ((temperatures[:, None, None] - means[None]) / (stds[None] + 1e-8)) ** 2)

total_dem = dems.sum(0) * dT
avg_T = (dems * temperatures[:, None, None]).sum(0) * dT

y = np.einsum('jkl,jm->mkl', dems * dT, k.cpu().numpy())  # 1e30 DN / s / px
image = np.array([norm(d) * 2 - 1 for d, norm in zip(y, sdo_norms.values())])
error = np.array([estimate_error(d * (u.ct / u.pix), c * u.AA).value / 1e2 for d, c in
                  zip(y, sdo_norms.keys())])
input_image = np.concatenate([image, error])

plt.subplot(121)
plt.imshow(total_dem)
plt.subplot(122)
plt.imshow(avg_T)
plt.savefig(os.path.join(evaluation_path, f'synthetic_dem.jpg'), dpi=300)
plt.close()

lambda_combinations = [[1e-4, 0, 0],]

for lambda_l1, lambda_l2, lambda_so in tqdm(lambda_combinations):
    # model = DEMModel(channel_response.shape[1], 3, T, k, normalization, n_dims=128)
    dem_model = DEMModule(model_path=model_path, lambda_l1=lambda_l1, lambda_l2=lambda_l2, lambda_so=lambda_so)
    # fit model
    dem_model.fit_images([input_image], batch_size=4)
    result = dem_model.compute(input_image, uncertainty=True)
    #
    image = (image + 1) / 2
    result['reconstruction'] = (result['reconstruction'] + 1) / 2
    print((np.abs(result['reconstruction'] - image).mean((1, 2)) / np.mean(image, (1, 2))) * 100)
    print(np.abs(result['reconstruction'] - image).mean((1, 2)))
    #
    ax.errorbar(T.cpu().numpy() * 1e-6, result['dem'].mean((1, 2)), yerr=result['dem_uncertainty'].mean((1, 2)), ecolor='red', capsize=2,
                label='Prediction')
    ax.plot(state['tempaxis'] * 1e-6, state['syntheticdem'], linestyle='--', label='GT')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 2e22)
    ax.set_xlabel('T [MK]')
    ax.set_ylabel('DEM')
    ax.set_title('')
    # ax.loglog()
    ax.legend()
    #
    fig.tight_layout()
    fig.savefig(os.path.join(evaluation_path, f'dem{lambda_l1:.0E}_{lambda_l2:.0E}_{lambda_so:.0E}.jpg'), dpi=300)
    plt.close(fig)

