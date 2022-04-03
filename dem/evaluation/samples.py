import datetime
import glob
import os
from random import randint, sample

from astropy.io.fits import PrimaryHDU, HDUList, ImageHDU
from matplotlib._color_data import TABLEAU_COLORS
from sunpy.map import Map, make_fitswcs_header
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from dateutil.parser import parse
from torch.utils.data import DataLoader

from dem.callback import sdo_cmaps
from dem.generator import DEMDataset, DEMMapDataset, prepMap

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u

base_dir = "/gss/r.jarolim/dem/version5"
temperate_response_path = '/gss/r.jarolim/data/aia_temperature_response.csv'

evaluation_path = os.path.join(base_dir, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

basenames_sdo = [[os.path.basename(f).split('_')[0] for f in glob.glob('/gss/r.jarolim/data/flare_prediction_v3/*_%s.fits' % wl)] for wl in ['94', '131', '171', '193', '211', '335']]
basenames_sdo = sorted(set(basenames_sdo[0]).intersection(*basenames_sdo[1:]))
dates_sdo = [parse(f) for f in basenames_sdo]

cond = [(d.month == 11) or (d.month == 12) for d in dates_sdo]
basenames_sdo = np.array(basenames_sdo)[cond]
sdo_files = [['/gss/r.jarolim/data/flare_prediction_v3/%s_%s.fits' % (bn, wl)
              for wl in ['94', '131', '171', '193', '211', '335']]
              for bn in basenames_sdo]
sdo_files = sample(sdo_files, 5)

sdo_valid_dataset = DEMMapDataset(sdo_files)

# load model
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
model = torch.load(os.path.join(base_dir, 'model.pt'))
model.eval()
model.to(device)


temperature_response = pd.read_csv(temperate_response_path).to_numpy()
temperatures = temperature_response[34:71, 0]

loader = DataLoader(sdo_valid_dataset, batch_size=1, shuffle=False)
with torch.no_grad():
    for images, file_cube in tqdm(zip(loader, sdo_files), total=len(loader)):
        start_time = datetime.datetime.now()
        log_dem_1 = model(images[:, :, :1024].to(device))[1].detach().cpu().numpy()
        log_dem_2 = model(images[:, :, 1024:].to(device))[1].detach().cpu().numpy()
        log_dem = np.concatenate([log_dem_1, log_dem_2], 2)
        for dem in log_dem:
            s_map = prepMap(Map(file_cube[0]))
            header = s_map.wcs.to_header()
            primary_hdu = PrimaryHDU(header=header)
            image_hdus = []
            for d, t in zip(dem, temperatures):
                header['LOG_TEMP'] = t
                image_hdus += [ImageHDU(d, header=header)]
            hdul = HDUList([primary_hdu, *image_hdus])
            hdul.writeto(os.path.join(evaluation_path, '%s.fits.gz' % s_map.date.to_datetime().isoformat('T')))
        print('Required Time:', datetime.datetime.now() - start_time)
