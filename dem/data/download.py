import argparse
import os
import shutil
from datetime import datetime, timedelta

import drms
from dateutil.parser import parse

parser = argparse.ArgumentParser()
parser.add_argument('--download_dir', type=str, required=True)
parser.add_argument('--email', type=str, required=True)
parser.add_argument('--t_start', type=str, required=True)
parser.add_argument('--duration', type=str, required=False, default=None)
parser.add_argument('--cadence', type=str, required=False, default='12s')
args = parser.parse_args()

download_dir = args.download_dir
wls = ['94', '131', '171', '193', '211', '335']
[os.makedirs(os.path.join(download_dir, wl), exist_ok=True) for wl in wls]

client = drms.Client(email=args.email, verbose=True)
start_date = args.t_start
r = client.export('aia.lev1_euv_12s[%s/%s@%s][%s]{image}' % (start_date, args.duration, args.cadence, ','.join(wls) ))
r.wait()


downloaded_files = r.download(download_dir)
for f in downloaded_files.download:
  path_elements = os.path.basename(f).split('.')
  f_date = path_elements[2]
  wl = path_elements[3]
  shutil.move(f, os.path.join(download_dir, wl, f_date[:-1] + '.fits'))