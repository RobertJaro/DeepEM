import glob
import os
from random import randint

from matplotlib._color_data import TABLEAU_COLORS
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from dateutil.parser import parse
from torch.utils.data import DataLoader

from dem.callback import sdo_cmaps
from dem.generator import DEMDataset

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

base_dir = "/gss/r.jarolim/dem/version5"
temperate_response_path = '/gss/r.jarolim/data/aia_temperature_response.csv'
converted_path = '/gss/r.jarolim/data/converted/flare_prediction_512_go_v3'

evaluation_path = os.path.join(base_dir, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

files = glob.glob(os.path.join(converted_path, '*.pickle'))
dates = [parse(os.path.basename(file).split('_')[1].replace('.pickle', '')) for file in files]
train_files = [f for f, d in zip(files, dates) if d.month in list(range(1, 11))]

valid_files = [f for f, d in zip(files, dates) if d.month in [11, 12]][:100]
dates = [d for d in dates if d.month in [11, 12]][:100]

sdo_valid_dataset = DEMDataset(valid_files, patch_shape=(512, 512))


# load model
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
model = torch.load(os.path.join(base_dir, 'model.pt'))
model.to(device)

temperature_response = pd.read_csv(temperate_response_path).to_numpy()
temperatures = temperature_response[34:71, 0]

loader = DataLoader(sdo_valid_dataset, batch_size=2, shuffle=False)
images = []
reconstructions = []
dems = []
with torch.no_grad():
    for image in tqdm(loader, total=len(loader)):
        reconstruction, log_dem = model(image.cuda())
        images.append(image.detach().cpu().numpy())
        reconstructions.append(reconstruction.detach().cpu().numpy())
        dems.append(log_dem.detach().cpu().numpy())

images = np.concatenate(images)
reconstructions = np.concatenate(reconstructions)
dems = np.concatenate(dems)

for idx in range(len(images)):

    img, rec, dem, date = images[idx], reconstructions[idx], dems[idx], dates[idx]

    points = [(randint(0, img.shape[1] - 1), randint(0, img.shape[2] - 1)) for _ in range(8)]
    colors = list(TABLEAU_COLORS.keys())[:8]

    fig, axs = plt.subplots(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(7, 7)
    axs.axis('off')

    for i, (c_img, cmap) in enumerate(zip(img, sdo_cmaps)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_axis_off()
        ax.imshow(c_img, cmap=cmap, vmin=-1, vmax=1)
        for p, color in zip(points, colors):
            ax.scatter(p[0], p[1], color=color)

    for i, (c_img, cmap) in enumerate(zip(rec, sdo_cmaps)):
        ax = fig.add_subplot(gs[1, i])
        ax.set_axis_off()
        ax.imshow(c_img, cmap=cmap, vmin=-1, vmax=1)

    dem_idx = np.linspace(0, len(dem) - 1, 7).astype(int)
    for i, (c_img, cmap) in enumerate(zip(dem[dem_idx], sdo_cmaps)):
        ax = fig.add_subplot(gs[2, i])
        ax.set_axis_off()
        ax.imshow(c_img, vmin=26, vmax=29)

    ax = fig.add_subplot(gs[3:, :])
    for p, color in zip(points, colors):
        ax.plot(temperatures, dem[:, p[0], p[1]], color=color)

    ax.plot(temperatures, np.mean(dem, axis=(1, 2)), color='black')
    ax.set_ylim(26, 29)
    ax.set_ylabel('DEM [log]')
    ax.set_xlabel('log T')

    fig.tight_layout()
    fig.savefig(os.path.join(evaluation_path, '%s.jpg') % date.isoformat('T'), dpi=80)
    plt.close(fig)
