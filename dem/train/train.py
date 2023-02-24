import argparse
import glob
import logging
import os
from random import randint

import numpy as np
import pandas as pd
import torch
from iti.data.dataset import StorageDataset
from iti.data.editor import BrightestPixelPatchEditor, LambdaEditor, RandomPatchEditor, Editor
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from dem.train.callback import PlotCallback
from dem.train.generator import AIADEMDataset, sdo_norms
from dem.train.model import DEMModel
from dem.module import DEMModule

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, required=True)
parser.add_argument('--temperature_response', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--converted_path', type=str, required=True)
parser.add_argument('--lambda_l1', type=float, required=False, default=1e-4)
parser.add_argument('--lambda_l2', type=float, required=False, default=0)
parser.add_argument('--lambda_so', type=float, required=False, default=0)
parser.add_argument('--n_dims', type=int, required=False, default=128)
args = parser.parse_args()

base_dir = args.base_dir
temperature_response_path = args.temperature_response

sdo_path = args.data_path
sdo_converted_path = args.converted_path

saturation_limit = 5
lambda_l1 = args.lambda_l1
lambda_l2 = args.lambda_l2
lambda_so = args.lambda_so

n_dims = args.n_dims

batch_size = 2 * torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1

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
logging.info(f'Configuration: L1 {lambda_l1}, L2 {lambda_l2}, Second-Order {lambda_so}')
logging.info(f'N_DIMs {n_dims}')

# init device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load temperature response
temperature_response = pd.read_csv(temperature_response_path).to_numpy()
t_filter = (temperature_response[:, 0] >= 5.7) & (temperature_response[:, 0] <= 7.4)
temperature_response = temperature_response[t_filter]
channel_response = temperature_response[:, 1:]
channel_response = np.delete(channel_response, 5, 1)  # remove 304
# interpolate
temperatures_original = temperature_response[:, 0]
temperatures = torch.arange(10 ** temperatures_original.min(), 10 ** temperatures_original.max(), 1e5)
channel_response = [np.interp(temperatures, 10 ** temperatures_original, channel_response[:, i])
                    for i in range(channel_response.shape[1])]
channel_response = np.stack(channel_response, 1)

# init tensors
k = torch.tensor(channel_response, dtype=torch.float32).to(device)
k[k < 0] = 0  # adjust invalid values
T = torch.tensor(np.array(temperatures), dtype=torch.float32).to(device)

normalization = torch.from_numpy(np.array([norm.vmax for norm in sdo_norms.values()])).float()

zero_weighting = -torch.log10(k.sum(1)).to(device)
zero_weighting = (zero_weighting - zero_weighting.min()) / (zero_weighting.max() - zero_weighting.min())

# init model
model = DEMModel(channel_response.shape[1], 3, T, k, normalization, n_dims=n_dims)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
dem_trainer = DEMModule(model=model, lambda_l1=lambda_l1, lambda_l2=lambda_l2, lambda_so=lambda_so)
logging.info('Model Size: %.03f M' % (sum(p.numel() for p in model.parameters()) / 1e6))

# init datasets
l_editor = LambdaEditor(lambda d: np.clip(d, a_min=-1, a_max=10, dtype=np.float32))
patch_shape = (256, 256)
train_patch_editor = BrightestPixelPatchEditor(patch_shape, random_selection=0.5)
valid_patch_editor = BrightestPixelPatchEditor(patch_shape, random_selection=0)

sdo_train_dataset = AIADEMDataset(sdo_path, patch_shape=(1024, 1024), months=[1, 2, 3, 4, 8, 9, 10, 11, 12])
sdo_train_dataset = StorageDataset(sdo_train_dataset, sdo_converted_path, ext_editors=[train_patch_editor, l_editor])

# sdo_train_dataset_random = AIADEMDataset(sdo_path, months=[1, 2, 3, 4, 8, 9, 10, 11, 12])
# sdo_train_dataset_random.addEditor(RandomPatchEditor([1024, 1024]))
# sdo_train_dataset_random = StorageDataset(sdo_train_dataset_random, sdo_converted_path + '_random',
#                                           ext_editors=[RandomPatchEditor(patch_shape), l_editor])

sdo_train_dataset = ConcatDataset([sdo_train_dataset] * 10) # 10 iterations per epoch

sdo_valid_dataset = AIADEMDataset(sdo_path, patch_shape=(1024, 1024), months=[5, 7])
sdo_valid_dataset = StorageDataset(sdo_valid_dataset, sdo_converted_path, ext_editors=[valid_patch_editor, l_editor])

#
sdo_train_loader = DataLoader(sdo_train_dataset, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
sdo_valid_loader = DataLoader(sdo_valid_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)

sdo_plot_dataset = AIADEMDataset(sdo_path, patch_shape=(1024, 1024), months=[5, 7], n_samples=8)
sdo_plot_dataset = StorageDataset(sdo_plot_dataset, sdo_converted_path, ext_editors=[valid_patch_editor, l_editor])
plot_callback = PlotCallback(sdo_plot_dataset, model, prediction_dir, temperatures.cpu().numpy(), saturation_limit,
                             device)

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


def percentage_diff(reconstructed_image, ref_image):
    ref_image, error = ref_image[:, :6], ref_image[:, 6:]
    saturation_mask = torch.min(ref_image < saturation_limit, dim=1, keepdim=True)[0]
    saturation_mask[saturation_mask == False] = float('nan')
    #
    reconstructed_image = (reconstructed_image + 1) / 2  # scale to [0, 1]
    sdo_image = (ref_image + 1) / 2  # scale to [0, 1]
    #
    loss = torch.nanmean(torch.abs(reconstructed_image - sdo_image) * saturation_mask, dim=(2, 3)) / \
           torch.nanmean(sdo_image, dim=(2, 3)) * 100
    loss = loss.mean(dim=0)
    return loss


for epoch in range(start_epoch, epochs):
    train_loss, zo_reg, so_reg = [], [], []
    dem_trainer.parallel_model.train()
    for iteration, sdo_image in enumerate(tqdm(sdo_train_loader, desc='Train')):
        sdo_image = sdo_image.to(device)
        optimizer.zero_grad()
        # forward
        reconstructed_image, dem = dem_trainer.parallel_model(sdo_image)
        # compute loss
        reconstruction_loss, l1_regularization, l2_regularization, so_regularization = dem_trainer._compute_loss(dem, reconstructed_image, sdo_image)
        # backward
        total_loss = reconstruction_loss + l1_regularization * dem_trainer.lambda_l1 + l2_regularization * dem_trainer.lambda_l2 + so_regularization * dem_trainer.lambda_so
        assert not torch.isnan(total_loss)
        total_loss.backward()
        optimizer.step()
        train_loss.append(reconstruction_loss.detach().cpu().numpy())
        zo_reg.append(l1_regularization.detach().cpu().numpy())
        so_reg.append(so_regularization.detach().cpu().numpy())
    #
    valid_loss = []
    valid_percentage = []
    dem_trainer.parallel_model.eval()
    with torch.no_grad():
        for sdo_image in tqdm(sdo_valid_loader, desc='Valid'):
            sdo_image = sdo_image.float().to(device)
            reconstructed_image, dem = dem_trainer.parallel_model(sdo_image)
            loss, l1_regularization, l2_regularization, so_regularization  = dem_trainer._compute_loss(dem, reconstructed_image, sdo_image)
            percentage = percentage_diff(reconstructed_image, sdo_image)
            valid_loss.append(loss.detach().cpu().numpy())
            valid_percentage.append(percentage.detach().cpu().numpy())
    # Add to history
    history['epoch'].append(epoch)
    history['train_loss'].append(np.mean(train_loss))
    history['valid_loss'].append(np.mean(valid_loss))
    history['zo_reg'].append(np.mean(zo_reg))
    history['so_reg'].append(np.mean(so_reg))
    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(6, 4))
    axs[0].plot(history['epoch'], history['train_loss'], 'o-', label='train loss')
    axs[0].plot(history['epoch'], history['valid_loss'], 'o-', label='valid loss')
    axs[0].set_ylim(None, min(max(history['valid_loss']), 1))
    axs[0].legend()
    axs[1].plot(history['epoch'], history['zo_reg'], 'o-', label='zeroth order regularization')
    axs[1].legend()
    axs[2].plot(history['epoch'], history['so_reg'], 'o-', label='second order regularization')
    axs[2].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(base_dir, 'history.jpg'))
    plt.close(fig)
    #
    plot_callback(epoch)
    # Logging
    logging.info(
        'EPOCH %04d/%04d [loss %.06f zero-reg %.06f second-reg %.06f] [valid loss %.06f] '
        '[channels 94: %.01f; 131: %.01f; 171: %.01f; 193: %.01f; 211: %.01f; 335: %.01f]' %
        (epoch + 1, epochs, np.mean(train_loss), np.mean(zo_reg), np.mean(so_reg), np.mean(valid_loss),
         *np.mean(valid_percentage, 0)))
    # Save
    torch.save({'m': model.state_dict(), 'o': optimizer.state_dict(),
                'epoch': epoch, 'history': history}, checkpoint_path)
    torch.save(model, model_path)
