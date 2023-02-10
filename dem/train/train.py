import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from iti.data.dataset import StorageDataset
from iti.data.editor import BrightestPixelPatchEditor, LambdaEditor, RandomPatchEditor
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from dem.train.callback import PlotCallback
from dem.train.generator import AIADEMDataset, sdo_norms
from dem.train.model import DEMModel, compute_weigted_loss

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, required=True)
parser.add_argument('--temperature_response', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--converted_path', type=str, required=True)
parser.add_argument('--lambda_zo', type=float, required=True, default=1e-4)
args = parser.parse_args()

base_dir = args.base_dir
temperate_response_path = args.temperature_response

sdo_path = args.data_path
sdo_converted_path = args.converted_path

saturation_limit = 5
lambda_zo = args.lambda_zo

n_dims = 512

batch_size = 3 * torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1

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
# zo_activation = nn.Softplus()
def zo_activation(x):
    x = torch.log10(x) - 20
    x = torch.tanh(x) + 1
    return x

temperature_response = pd.read_csv(temperate_response_path).to_numpy()
# select smaller temperature range
t_filter = (temperature_response[:, 0] >= 5.7) & (temperature_response[:,
                                                  0] <= 7.4)  # (temperature_response[:, 0] >= 5.5) & (temperature_response[:, 0] <= 7.4)
temperature_response = temperature_response[t_filter]

channel_response = temperature_response[:, 1:]
channel_response = np.delete(channel_response, 5, 1)  # remove 304
k = torch.from_numpy(channel_response).float()
k[k < 0] = 0  # adjust invalid values

temperatures = 10 ** torch.tensor(temperature_response[:, 0], dtype=torch.float32).to(device)

l_editor = LambdaEditor(lambda d: np.clip(d, a_min=-1, a_max=10, dtype=np.float32))
train_patch_editor = BrightestPixelPatchEditor((128, 128), random_selection=0.5)
valid_patch_editor = BrightestPixelPatchEditor((128, 128), random_selection=0)

sdo_train_dataset = AIADEMDataset(sdo_path, patch_shape=(1024, 1024), months=[1, 2, 3, 4, 8, 9, 10, 11, 12])
sdo_train_dataset = StorageDataset(sdo_train_dataset, sdo_converted_path, ext_editors=[train_patch_editor, l_editor])

# TODO quick check
sdo_train_dataset_random = AIADEMDataset(sdo_path, months=[1, 2, 3, 4, 8, 9, 10, 11, 12])
sdo_train_dataset_random.addEditor(RandomPatchEditor([1024, 1024]))
sdo_train_dataset_random = StorageDataset(sdo_train_dataset_random, sdo_converted_path + '_random', ext_editors=[RandomPatchEditor([128, 128]), l_editor])

sdo_train_dataset = ConcatDataset([sdo_train_dataset, sdo_train_dataset_random])

sdo_valid_dataset = AIADEMDataset(sdo_path, patch_shape=(1024, 1024), months=[5, 7])
sdo_valid_dataset = StorageDataset(sdo_valid_dataset, sdo_converted_path, ext_editors=[valid_patch_editor, l_editor])

normalization = torch.from_numpy(np.array([norm.vmax for norm in sdo_norms.values()])).float()
zero_weighting = -torch.log10(k.sum(1)).to(device)
zero_weighting = (zero_weighting - zero_weighting.min()) / (zero_weighting.max() - zero_weighting.min())

# plot weighting
plt.figure(figsize=(4, 2))
plt.plot(temperatures.cpu(), zero_weighting.cpu())
plt.xlim(0, 25e6)
plt.savefig(os.path.join(prediction_dir, 'zero_weighting.jpg'))
plt.close()
# Init Model
model = DEMModel(channel_response.shape[1], 10, temperature_response[:, 0], k, normalization, n_dims=n_dims, saturation_limit=saturation_limit)
logging.info('Model Size: %.03f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
parallel_model = nn.DataParallel(model)
parallel_model.to(device)

optimizer = optim.Adam(parallel_model.parameters(), lr=1e-5)

#
sdo_train_loader = DataLoader(sdo_train_dataset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)
sdo_valid_loader = DataLoader(sdo_valid_dataset, batch_size=batch_size, num_workers=os.cpu_count() // 2)

sdo_plot_dataset = AIADEMDataset(sdo_path, patch_shape=(1024, 1024), months=[5, 7], n_samples=8)
sdo_plot_dataset = StorageDataset(sdo_plot_dataset, sdo_converted_path, ext_editors=[valid_patch_editor, l_editor])
plot_callback = PlotCallback(sdo_plot_dataset, parallel_model, prediction_dir, temperatures.cpu().numpy(),
                             saturation_limit, device)

start_epoch = 0
history = {'epoch': [], 'train_loss': [], 'valid_loss': [], 'so_reg': [], 'zo_reg': []}
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['m'])
    optimizer.load_state_dict(state_dict['o'])
    start_epoch = state_dict['epoch'] + 1
    history = state_dict['history']

# history = {'epoch': [], 'train_loss': [], 'valid_loss': [], 'so_reg':[], 'zo_reg':[]}
plot_callback(-1)
# Start training
epochs = 1000

# channel_weighting = torch.tensor([5, 1, 1, 0.5, 0.5, 5], dtype=torch.float32).view(1, 6, 1, 1).to(device)
stretch_div = torch.arcsinh(torch.tensor(1 / 0.005))


def percentage_diff(reconstructed_image, sdo_image):
    saturation_mask = torch.min(sdo_image < saturation_limit, dim=1, keepdim=True)[0]
    saturation_mask[saturation_mask == False] = float('nan')
    #
    reconstructed_image = (reconstructed_image + 1) / 2  # scale to [0, 1]
    sdo_image = (sdo_image + 1) / 2  # scale to [0, 1]
    #
    loss = torch.nanmean(torch.abs(reconstructed_image - sdo_image) * saturation_mask, dim=(0, 2, 3)) / torch.nanmean(
        sdo_image, dim=(0, 2, 3)) * 100
    return loss


for epoch in range(start_epoch, epochs):
    parallel_model.train()
    train_loss = []
    zo_reg = []
    for iteration, sdo_image in enumerate(tqdm(sdo_train_loader, desc='Train')):
        sdo_image = sdo_image.to(device)
        optimizer.zero_grad()
        reconstructed_image, dem = parallel_model(sdo_image)

        loss = compute_weigted_loss(reconstructed_image, sdo_image)
        saturation_mask = torch.min(sdo_image < saturation_limit, dim=1)[0][:, None]
        # clip_mask = dem < 1e23 # only regularize small values - ignore large values
        # zeroth_order_regularization = torch.sum(dem / 1e23 * saturation_mask, dim=1).mean()
        zeroth_order_regularization = torch.mean(zo_activation(dem) * saturation_mask, dim=1).mean()

        total_loss = loss + zeroth_order_regularization * lambda_zo
        assert not torch.isnan(total_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parallel_model.parameters(), 0.1)
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
        zo_reg.append(zeroth_order_regularization.detach().cpu().numpy())
    #
    valid_loss = []
    valid_percentage = []
    parallel_model.eval()
    with torch.no_grad():
        for sdo_image in tqdm(sdo_valid_loader, desc='Valid'):
            sdo_image = sdo_image.float().to(device)
            reconstructed_image, dem = parallel_model(sdo_image)
            loss = compute_weigted_loss(reconstructed_image, sdo_image)
            percentage = percentage_diff(reconstructed_image, sdo_image)
            valid_loss.append(loss.detach().cpu().numpy())
            valid_percentage.append(percentage.detach().cpu().numpy())
    # Add to history
    history['epoch'].append(epoch)
    history['train_loss'].append(np.mean(train_loss))
    history['valid_loss'].append(np.mean(valid_loss))
    history['zo_reg'].append(np.mean(zo_reg))
    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(6, 4))
    axs[0].plot(history['epoch'], history['train_loss'], 'o-', label='train loss')
    axs[0].plot(history['epoch'], history['valid_loss'], 'o-', label='valid loss')
    axs[0].set_ylim(None, min(max(history['valid_loss']), 1))
    axs[0].legend()
    axs[1].plot(history['epoch'], history['zo_reg'], 'o-', label='zeroth order regularization')
    axs[1].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(base_dir, 'history.jpg'))
    plt.close(fig)
    #
    plot_callback(epoch)
    # Logging
    logging.info(
        'EPOCH %04d/%04d [loss %.06f zero-reg %.06f] [valid loss %.06f] '
        '[channels 94: %.01f; 131: %.01f; 171: %.01f; 193: %.01f; 211: %.01f; 335: %.01f]' %
        (epoch + 1, epochs, np.mean(train_loss), np.mean(zo_reg), np.mean(valid_loss), *np.mean(valid_percentage, 0)))
    # Save
    torch.save({'m': model.state_dict(), 'o': optimizer.state_dict(),
                'epoch': epoch, 'history': history}, checkpoint_path)
    torch.save(model, model_path)
