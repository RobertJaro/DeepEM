import glob
import os

import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch
from iti.data.editor import NormalizeRadiusEditor
from sunpy.coordinates import frames
from sunpy.map import make_fitswcs_header, Map
from sunpy.visualization.colormaps import cm

out_path = '/gpfs/gpfs0/robert.jarolim/data/dem/simulation_prep'
os.makedirs(out_path, exist_ok=True)

for file in glob.glob('/gpfs/gpfs0/robert.jarolim/data/dem/simulation/*.fits'):
    prep_file = f'{out_path}/{os.path.basename(file)}'
    if os.path.exists(prep_file):
        os.remove(prep_file)
    data = fits.getdata(file)

    coordinate = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime="2009-02-13",
                          observer='earth', frame=frames.Helioprojective)
    header = make_fitswcs_header(data, coordinate=coordinate, scale=u.Quantity([0.6, 0.6] * u.arcsec / u.pix))

    s_map = Map(data, header)
    cc = SkyCoord(-400 * u.arcsec, -200 * u.arcsec, frame=s_map.coordinate_frame)
    c_pix = s_map.world_to_pixel(cc)
    bl_pix = u.Quantity((c_pix.x - 512 * u.pix, c_pix.y - 512 * u.pix))
    print(bl_pix)
    sub_map = s_map.submap(bottom_left=bl_pix, width=1023 * u.pix, height=1023 * u.pix)
    print(sub_map.data.shape)
    sub_map.save(prep_file)

s_map = Map(os.path.join(out_path, '193A_Evenorm.fits'))
s_map.plot(norm=ImageNormalize(vmin=0, vmax=1e4, stretch=AsinhStretch(0.005)), cmap=cm.sdoaia193)
plt.savefig('/gpfs/gpfs0/robert.jarolim/dem/uc_version1/evaluation_sim/demo.jpg')
plt.close()

