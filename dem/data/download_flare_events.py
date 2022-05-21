import argparse
import os
import random
import shutil
from datetime import datetime, timedelta

import drms
import pandas as pd
from dateutil.parser import parse

parser = argparse.ArgumentParser()
parser.add_argument('--download_dir', type=str, required=True)
parser.add_argument('--flare_list', type=str, required=True)
parser.add_argument('--email', type=str, required=True)
args = parser.parse_args()

download_dir = args.download_dir
flare_list = pd.read_csv(args.flare_list, parse_dates=['start_time', 'end_time'])
wls = ['94', '131', '171', '193', '211', '335']
[os.makedirs(os.path.join(download_dir, wl), exist_ok=True) for wl in wls]

client = drms.Client(email=args.email, verbose=True)
for i, flare in flare_list.iterrows():
  start_date = flare['start_time'].to_pydatetime()
  end_date = flare['end_time'].to_pydatetime()
  flare_dates = [start_date + i * timedelta(minutes=1) for i in range((end_date - start_date) // timedelta(minutes=1))]
  date = random.sample(flare_dates, 1)[0]
  r = client.export('aia.lev1_euv_12s[%s][%s]{image}' % (date.isoformat('T'), ','.join(wls) ))
  r.wait()

  downloaded_files = r.download(download_dir)
  for f in downloaded_files.download:
    path_elements = os.path.basename(f).split('.')
    f_date = path_elements[2]
    wl = path_elements[3]
    shutil.move(f, os.path.join(download_dir, wl, f_date[:-1] + '.fits'))