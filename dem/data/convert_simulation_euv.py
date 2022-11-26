import glob
import os

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from iti.data.editor import NormalizeRadiusEditor
from sunpy.coordinates import frames
from sunpy.map import make_fitswcs_header, Map

out_path = '/gpfs/gpfs0/robert.jarolim/data/dem/simulation_prep'
os.makedirs(out_path, exist_ok=True)

for file in glob.glob('/gpfs/gpfs0/robert.jarolim/data/dem/simulation/*.fits'):
    prep_file = f'{out_path}/{os.path.basename(file)}'
    if os.path.exists(prep_file):
        continue
    data = fits.getdata(file)

    coordinate = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime="2009-02-13",
                          observer='earth', frame=frames.Helioprojective)
    header = make_fitswcs_header(data, coordinate=coordinate, scale=u.Quantity([0.6, 0.6] * u.arcsec / u.pix))

    s_map = Map(data, header)
    s_map.save(prep_file)
