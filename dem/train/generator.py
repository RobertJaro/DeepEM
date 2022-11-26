import numpy as np
from aiapy.calibrate import register, normalize_exposure, correct_degradation
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import ImageNormalize, LinearStretch
from iti.data.dataset import StackDataset, get_intersecting_files, BaseDataset
from iti.data.editor import BrightestPixelPatchEditor, LoadMapEditor, AIAPrepEditor, \
    MapToDataEditor, NormalizeEditor, LambdaEditor, ExpandDimsEditor, Editor
from sunpy.map import Map

sdo_norms = {94: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             131: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             171: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             193: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             211: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             335: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             }


class PrepEditor(Editor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, s_map, **kwargs):
        s_map = register(s_map)
        s_map = correct_degradation(s_map)
        s_map = normalize_exposure(s_map)
        # pad if required
        pad_x = 4096 - s_map.data.shape[1]
        pad_y = 4096 - s_map.data.shape[0]
        data = np.pad(s_map.data, ([np.floor(pad_y / 2).astype(np.int), np.ceil(pad_y / 2).astype(np.int)],
                                   [np.floor(pad_x / 2).astype(np.int), np.ceil(pad_x / 2).astype(np.int)],),
                      constant_values=0)
        s_map = Map(data, s_map.meta)
        return s_map

class LinearAIADataset(BaseDataset):

    def __init__(self, data, wavelength, ext='.fits', **kwargs):
        norm = sdo_norms[wavelength]

        editors = [LoadMapEditor(),
                   PrepEditor(),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ExpandDimsEditor(),
                   LambdaEditor(lambda d, **_: np.clip(d, a_min=-1, a_max=10, dtype=np.float32))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)


class DEMDataset(StackDataset):

    def __init__(self, data, patch_shape=None, ext='.fits', **kwargs):
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, ['94', '131', '171', '193', '211', '335'], ext=ext, **kwargs)
        data_sets = [LinearAIADataset(paths[0], 94),
                     LinearAIADataset(paths[1], 131),
                     LinearAIADataset(paths[2], 171),
                     LinearAIADataset(paths[3], 193),
                     LinearAIADataset(paths[4], 211),
                     LinearAIADataset(paths[5], 335)
                     ]
        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape, random_selection=0))

class FITSDataset(BaseDataset):

    def __init__(self, data, norm, ext='.fits', **kwargs):


        editors = [LoadMapEditor(),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ExpandDimsEditor(),
                   LambdaEditor(lambda d, **_: np.clip(d, a_min=-1, a_max=10, dtype=np.float32))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)

class FITSDEMDataset(StackDataset):

    def __init__(self, data, patch_shape=None, ext='.fits', **kwargs):
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, ['94', '131', '171', '193', '211', '335'], ext=ext, **kwargs)
        data_sets = [FITSDataset(paths[0], sdo_norms[94]),
                     FITSDataset(paths[1], sdo_norms[131]),
                     FITSDataset(paths[2], sdo_norms[171]),
                     FITSDataset(paths[3], sdo_norms[193]),
                     FITSDataset(paths[4], sdo_norms[211]),
                     FITSDataset(paths[5], sdo_norms[335])
                     ]
        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape, random_selection=0))


def prep_aia_map(s_map, calibrate=True):
    norm = sdo_norms[int(s_map.wavelength.value)]
    if calibrate:
        aia_prep_editor = AIAPrepEditor('aiapy')
        s_map = aia_prep_editor.call(s_map)
    data = norm(s_map.data).astype(np.float32) * 2 - 1
    data = np.clip(data, a_min=-1, a_max=10, dtype=np.float32)
    s_map = Map(data, s_map.meta)
    return s_map