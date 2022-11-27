import argparse
import os
import shutil

import drms

parser = argparse.ArgumentParser()
parser.add_argument('--download_dir', type=str, required=True)
parser.add_argument('--email', type=str, required=True)
args = parser.parse_args()

client = drms.Client(verbose=True)

# Download directory
download_dir = args.download_dir
wls = ['94', '131', '171', '193', '211', '335']
[os.makedirs(os.path.join(download_dir, wl), exist_ok=True) for wl in wls]

# email
email = args.email

process = {
    "im_patch": {
        "t_ref": "2020-06-07T21:44:28",
        "t": 0,
        "r": 0,
        "c": 0,
        "locunits": "arcsec",
        "boxunits": "pixels",
        "x": -335,
        "y": -385,
        "width": 256,
        "height": 256,
    }
}

# Submit export request using the 'fits' protocol
qstr = "aia.lev1_euv_12s[2020-06-07T21:30:00/60m@12s][94, 131, 171, 193, 211, 335]{image}"

result = client.export(
    qstr,
    method="url",
    protocol="fits",
    email=email,
    process=process,
)

# Download selected files.
result.wait()
downloaded_files = result.download(download_dir)
for f in downloaded_files.download:
    path_elements = os.path.basename(f).split('.')
    f_date = path_elements[2]
    wl = path_elements[3]
    shutil.move(f, os.path.join(download_dir, wl, f_date[:-1] + '.fits'))