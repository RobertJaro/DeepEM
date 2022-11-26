import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map

from dem.train.generator import PrepEditor

download_dir = '/gpfs/gpfs0/robert.jarolim/data/dem_test'
wls = ['94', '131', '171', '193', '211', '335']

files = [sorted(glob.glob(os.path.join(download_dir, wl, '*.fits'))) for wl in wls]
files = np.concatenate(files)

prep_editor = PrepEditor()

s_map = Map(files[0])
s_map = prep_editor.call(s_map)
#
center = SkyCoord(-335 * u.arcsec, -385 * u.arcsec, frame=s_map.coordinate_frame)
center = center.transform_to(frames.HeliographicCarrington)

# save maps
for f in files:
    s_map = Map(f)
    s_map = prep_editor.call(s_map)
    # use same carrington center coord
    cc = SkyCoord(center.lon, center.lat, frame=frames.HeliographicCarrington, observer=s_map.observer_coordinate)
    c_pix = s_map.world_to_pixel(cc)
    bl_pix = u.Quantity((c_pix.x - 128 * u.pix, c_pix.y - 128 * u.pix))
    print(bl_pix)
    sub_map = s_map.submap(bottom_left=bl_pix, width=255 * u.pix, height=255 * u.pix)
    new_path = f.replace('dem_test', 'dem_test_prep')
    os.makedirs(os.path.split(new_path)[0], exist_ok=True)
    if os.path.exists(new_path):
        os.remove(new_path)
    sub_map.save(new_path)
    #
    print(sub_map.data.shape)
    sub_map.plot()
    plt.savefig(f'/gpfs/gpfs0/robert.jarolim/data/dem_test_prep/{os.path.basename(f).replace("fits", "jpg")}')
    plt.close()

shutil.make_archive('/gpfs/gpfs0/robert.jarolim/data/dem_test_prep', 'zip',
                    '/gpfs/gpfs0/robert.jarolim/data/dem_test_prep')
