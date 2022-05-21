import argparse
import logging
import os

from iti.data.dataset import StorageDataset
from iti.data.editor import BrightestPixelPatchEditor, LambdaEditor
from torch import optim
from torch.nn import MSELoss
from tqdm import tqdm

from dem.train.callback import PlotCallback
from dem.train.generator import DEMDataset, sdo_norms
from dem.train.model import DeepEMModel

import torch
from torch.utils.data import DataLoader, RandomSampler

from matplotlib import pyplot as plt
import pandas as pd

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
parser.add_argument('--num_workers', type=int, required=False, default=4)
parser.add_argument('--base_dir', type=str, required=True)
parser.add_argument('--temperature_response', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--converted_path', type=str, required=True)
args = parser.parse_args()

base_dir = args.base_dir
temperate_response_path = args.temperature_response

sdo_path = args.data_path
sdo_converted_path = args.converted_path

lambda_zo = 1e-4
lambda_so = 1e-2

prediction_dir = os.path.join(base_dir, 'prediction')
os.makedirs(prediction_dir, exist_ok=True)
checkpoint_path = os.path.join(base_dir, 'checkpoint.pt')
model_path = os.path.join(base_dir, 'model.pt')

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

temperature_response = pd.read_csv(temperate_response_path).to_numpy()
temperature_response = temperature_response[:, 1:]
temperature_response = np.delete(temperature_response, 5, 1)  # remove 304
k = torch.from_numpy(temperature_response).float()

l_editor = LambdaEditor(lambda d: np.clip(d, a_min=-1, a_max=None, dtype=np.float32))
patch_editor = BrightestPixelPatchEditor((256, 256), random_selection=0.5)

sdo_train_dataset = DEMDataset(sdo_path, patch_shape=(1024, 1024), resolution=4096, months=list(range(11)))
sdo_train_dataset = StorageDataset(sdo_train_dataset, sdo_converted_path, ext_editors=[patch_editor, l_editor])

sdo_valid_dataset = DEMDataset(sdo_path, patch_shape=(1024, 1024), resolution=4096, months=[11, 12])
sdo_valid_dataset = StorageDataset(sdo_valid_dataset, sdo_converted_path, ext_editors=[patch_editor, l_editor])

normalization = torch.from_numpy(np.array([norm.vmax for norm in sdo_norms.values()])).float()
zero_weighting = (k.sum(1) / 1e-30).to(device)
# Init Model
model = DeepEMModel(temperature_response.shape[1], temperature_response.shape[0], k, normalization)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-5)

#
sdo_train_loader = DataLoader(sdo_train_dataset, batch_size=6, num_workers=4, sampler=RandomSampler(sdo_train_dataset, replacement=True, num_samples=160))
sdo_valid_loader = DataLoader(sdo_valid_dataset, batch_size=12, shuffle=True, num_workers=4)

sdo_plot_dataset = DEMDataset(sdo_path, patch_shape=(1024, 1024), resolution=4096, months=[11, 12], n_samples=8)
sdo_plot_dataset = StorageDataset(sdo_plot_dataset, sdo_converted_path, ext_editors=[patch_editor, l_editor])
plot_callback = PlotCallback(sdo_plot_dataset, model, prediction_dir)
plot_callback(-1)

start_epoch = 0
history = {'epoch': [], 'train_loss': [], 'valid_loss': []}
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['m'])
    optimizer.load_state_dict(state_dict['o'])
    start_epoch = state_dict['epoch'] + 1
    history = state_dict['history']

criterion = MSELoss()

# Start training
epochs = 2000
for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = []
    so_reg = []
    zo_reg = []
    for iteration, sdo_image in enumerate(tqdm(sdo_train_loader, desc='Train')):
        sdo_image = sdo_image.to(device)
        optimizer.zero_grad()
        reconstructed_image, dem = model(sdo_image)

        loss = criterion(reconstructed_image, sdo_image)
        second_order_regularization = (dem[:, :-2] - 2 * dem[:, 1:-1] + dem[:, 2:]).pow(2).mean()
        clipped_dem = torch.clip(dem - 25, min=0)
        zeroth_order_regularization = torch.sum(clipped_dem ** 2 / zero_weighting[None, :, None, None], dim=1).pow(0.5).mean()

        total_loss = loss + second_order_regularization * lambda_so + zeroth_order_regularization * lambda_zo
        assert not torch.isnan(total_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
        so_reg.append(second_order_regularization.detach().cpu().numpy())
        zo_reg.append(zeroth_order_regularization.detach().cpu().numpy())
    valid_loss = []
    model.eval()
    with torch.no_grad():
        for sdo_image in tqdm(sdo_valid_loader, desc='Valid'):
            sdo_image = sdo_image.float().to(device)
            reconstructed_image, dem = model(sdo_image)
            loss = criterion(reconstructed_image, sdo_image)
            valid_loss.append(loss.detach().cpu().numpy())
    # Add to history
    history['epoch'].append(epoch)
    history['train_loss'].append(np.mean(train_loss))
    history['valid_loss'].append(np.mean(valid_loss))
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(history['epoch'], history['train_loss'], 'o-', label='train loss')
    ax.plot(history['epoch'], history['valid_loss'], 'o-', label='valid loss')
    ax.legend()
    fig.savefig(os.path.join(base_dir, 'history.jpg'))
    plt.close(fig)
    #
    plot_callback(epoch)
    # Logging
    logging.info(
        'EPOCH %04d/%04d [loss %.06f second-reg %.06f zero-reg %.06f] [valid loss %.06f]' %
        (epoch + 1, epochs, np.mean(train_loss), np.mean(so_reg), np.mean(zo_reg), np.mean(valid_loss)))
    # Save
    torch.save({'m': model.state_dict(), 'o': optimizer.state_dict(),
                'epoch': epoch, 'history': history}, checkpoint_path)
    torch.save(model, model_path)
