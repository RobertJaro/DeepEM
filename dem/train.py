import logging
import os

from iti.data.dataset import StorageDataset
from iti.data.editor import RandomPatchEditor, BrightestPixelPatchEditor
from torch import optim
from torch.nn import MSELoss
from tqdm import tqdm

from dem.callback import PlotCallback
from dem.generator import DEMDataset
from dem.model import DeepEM, TVLoss

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import pandas as pd

import numpy as np

base_dir = "/gss/r.jarolim/dem/version7"
temperate_response_path = '/gss/r.jarolim/data/aia_temperature_response.csv'
sdo_path = "/gss/r.jarolim/data/ch_detection"
sdo_converted_path = '/gss/r.jarolim/data/converted/dem_4096'

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
temperature_response = temperature_response[34:71, 1:]
temperature_response = np.delete(temperature_response, 5, 1)  # remove 304
k = torch.from_numpy(temperature_response).float().to(device)

patch_editor = BrightestPixelPatchEditor((256, 256), random_selection=0.5)

sdo_train_dataset = DEMDataset(sdo_path, patch_shape=(1024, 1024), resolution=4096, months=list(range(11)), limit=100)
sdo_train_dataset = StorageDataset(sdo_train_dataset, sdo_converted_path, ext_editors=[patch_editor])

sdo_valid_dataset = DEMDataset(sdo_path, patch_shape=(1024, 1024), resolution=4096, months=[11, 12], limit=100)
sdo_valid_dataset = StorageDataset(sdo_valid_dataset, sdo_converted_path, ext_editors=[patch_editor])

normalization = torch.from_numpy(np.array([340, 1400, 8600, 9800, 5800, 600])).float().to(device)
# Init Model
model = DeepEM(6, 37, k, normalization)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

#
sdo_train_loader = DataLoader(sdo_train_dataset, batch_size=8, shuffle=True, num_workers=16)
sdo_valid_loader = DataLoader(sdo_valid_dataset, batch_size=16, shuffle=True, num_workers=16)

sdo_plot_dataset = DEMDataset(sdo_path, patch_shape=(1024, 1024), resolution=4096, months=[11, 12], n_samples=8)
sdo_plot_dataset = StorageDataset(sdo_plot_dataset, sdo_converted_path, ext_editors=[patch_editor])
plot_callback = PlotCallback(sdo_plot_dataset, model, prediction_dir)
plot_callback(0)

start_epoch = 0
history = {'epoch': [], 'train_loss': [], 'valid_loss': []}
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['m'])
    optimizer.load_state_dict(state_dict['o'])
    start_epoch = state_dict['epoch'] + 1
    history = state_dict['history']

criterion = MSELoss()
tv_criterion = TVLoss(weight=0.001)

# Start training
epochs = 100
for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = []
    for iteration, batch in enumerate(tqdm(sdo_train_loader, desc='Train')):
        batch = batch.float().to(device)
        optimizer.zero_grad()
        reconstructed_batch, dem = model(batch)
        loss = criterion(reconstructed_batch, batch)
        tv_loss = tv_criterion(dem)
        minimum_loss = torch.mean(dem) * 0.0001 # TODO /k --> press to 0 where already small response; crop large values
        total_loss = loss + tv_loss + minimum_loss
        total_loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    valid_loss = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(sdo_valid_loader, desc='Valid'):
            batch = batch.float().to(device)
            reconstructed_batch, dem = model(batch)
            loss = criterion(reconstructed_batch, batch)
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
        'EPOCH %04d/%04d [loss %.06f] [valid loss %.06f]' % (epoch, epochs, np.mean(train_loss), np.mean(valid_loss)))
    # Save
    torch.save({'m': model.state_dict(), 'o': optimizer.state_dict(),
                'epoch': epoch, 'history': history}, checkpoint_path)
    torch.save(model, model_path)
