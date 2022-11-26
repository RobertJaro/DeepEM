import os

import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
from scipy.io import readsav
from sunpy.coordinates import frames
from sunpy.map import make_fitswcs_header, Map

data_path = '/Volumes/Extreme SSD/IDL_MAS_2021Eclipse_Cosie_3D_EM.sav'
dem3d_data = readsav(data_path)['dem3d']

out_path = '/Volumes/Extreme SSD/DEM'
os.makedirs(out_path, exist_ok=True)

norm = ImageNormalize(vmin=0, stretch=AsinhStretch(0.005))

for i, data in enumerate(dem3d_data):
    prep_file = os.path.join(out_path, 'simulation_T%02d.fits' % i)
    # if os.path.exists(prep_file):
    #     continue

    coordinate = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime="2009-02-13",
                          observer='earth', frame=frames.Helioprojective)
    header = make_fitswcs_header(data, coordinate=coordinate, scale=u.Quantity([3.17, 3.17] * u.arcsec / u.pix))

    s_map = Map(data, header)

    arcs_frame = 1500 * u.arcsec
    s_map = s_map.submap(
        bottom_left=SkyCoord(-arcs_frame, -arcs_frame, frame=s_map.coordinate_frame),
        top_right=SkyCoord(arcs_frame, arcs_frame, frame=s_map.coordinate_frame))

    scale_factor = 3.17 / 0.6
    s_map = s_map.rotate(recenter=False, scale=scale_factor, missing=0, order=4)
    center_pix = s_map.data.shape[0] / 2
    s_map = s_map.submap(bottom_left=[center_pix - 2048, center_pix - 2048] * u.pix,
                         width=4095 * u.pix, height=4095 * u.pix)
    s_map.save(prep_file)

test_map = Map('/Volumes/Extreme SSD/DEM/simulation_T01.fits')
test_map.plot(norm=norm)
test_map.draw_limb()
plt.show()
