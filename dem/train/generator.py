import numpy as np
from aiapy.calibrate import register, normalize_exposure, correct_degradation, estimate_error
from astropy.visualization import ImageNormalize, LinearStretch, AsinhStretch
from iti.data.dataset import StackDataset, get_intersecting_files, BaseDataset
from iti.data.editor import BrightestPixelPatchEditor, LoadMapEditor, AIAPrepEditor, \
    MapToDataEditor, NormalizeEditor, LambdaEditor, ExpandDimsEditor, Editor, get_local_correction_table
from sunpy.map import Map

from astropy import units as u

sdo_norms = {94:  ImageNormalize(vmin=0, vmax=1e4, stretch=AsinhStretch(0.005), clip=False),
             131: ImageNormalize(vmin=0, vmax=1e4, stretch=AsinhStretch(0.005), clip=False),
             171: ImageNormalize(vmin=0, vmax=1e4, stretch=AsinhStretch(0.005), clip=False),
             193: ImageNormalize(vmin=0, vmax=1e4, stretch=AsinhStretch(0.005), clip=False),
             211: ImageNormalize(vmin=0, vmax=1e4, stretch=AsinhStretch(0.005), clip=False),
             335: ImageNormalize(vmin=0, vmax=1e4, stretch=AsinhStretch(0.005), clip=False),
             }


class PrepEditor(Editor):
    def __init__(self, skip_register=False, normalize_exposure= True, **kwargs):
        super().__init__(**kwargs)
        self.skip_register = skip_register
        self.normalize_exposure = normalize_exposure
        self.table = get_local_correction_table()

    def call(self, s_map, **kwargs):
        s_map = correct_degradation(s_map, correction_table=self.table)
        s_map = normalize_exposure(s_map) if self.normalize_exposure else s_map

        if self.skip_register:
            return s_map

        s_map = register(s_map)
        # pad if required
        pad_x = 4096 - s_map.data.shape[1]
        pad_y = 4096 - s_map.data.shape[0]
        data = np.pad(s_map.data, ([np.floor(pad_y / 2).astype(np.int), np.ceil(pad_y / 2).astype(np.int)],
                                   [np.floor(pad_x / 2).astype(np.int), np.ceil(pad_x / 2).astype(np.int)],),
                      constant_values=0)
        s_map = Map(data, s_map.meta)
        return s_map

class ErrorEditor(Editor):

    def __init__(self, wavelength, scaling_factor=1e2, **kwargs):
        super().__init__(**kwargs)
        self.wavelength = wavelength
        self.scaling_factor = scaling_factor

    def call(self, data, **kwargs):
        error = estimate_error(data * (u.ct / u.pix), self.wavelength * u.AA).value
        return np.nan_to_num(error / self.scaling_factor, nan=1)

class LinearAIADataset(BaseDataset):

    def __init__(self, data, wavelength, skip_register=False, ext='.fits', **kwargs):
        norm = sdo_norms[wavelength]

        editors = [LoadMapEditor(),
                   PrepEditor(skip_register),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ExpandDimsEditor(),
                   LambdaEditor(lambda d, **_: np.clip(d, a_min=-1, a_max=10, dtype=np.float32))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)

class ErrorDataset(BaseDataset):

    def __init__(self, data, wavelength, skip_register=False, skip_prep=False, ext='.fits', **kwargs):
        editors = [LoadMapEditor(),
                   PrepEditor(skip_register, normalize_exposure=False),
                   MapToDataEditor(),
                   ErrorEditor(wavelength),
                   ExpandDimsEditor(),]
        if skip_prep:
            del editors[1]
        super().__init__(data, editors=editors, ext=ext, **kwargs)

class AIADEMDataset(StackDataset):

    def __init__(self, data, patch_shape=None, skip_register=False, ext='.fits', **kwargs):
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, ['94', '131', '171', '193', '211', '335'], ext=ext, **kwargs)
        data_sets = [LinearAIADataset(paths[0], 94, skip_register=skip_register),
                     LinearAIADataset(paths[1], 131, skip_register=skip_register),
                     LinearAIADataset(paths[2], 171, skip_register=skip_register),
                     LinearAIADataset(paths[3], 193, skip_register=skip_register),
                     LinearAIADataset(paths[4], 211, skip_register=skip_register),
                     LinearAIADataset(paths[5], 335, skip_register=skip_register),
                     ErrorDataset(paths[0], 94, skip_register=skip_register),
                     ErrorDataset(paths[1], 131, skip_register=skip_register),
                     ErrorDataset(paths[2], 171, skip_register=skip_register),
                     ErrorDataset(paths[3], 193, skip_register=skip_register),
                     ErrorDataset(paths[4], 211, skip_register=skip_register),
                     ErrorDataset(paths[5], 335, skip_register=skip_register)
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
                     FITSDataset(paths[5], sdo_norms[335]),
                     ErrorDataset(paths[0], 94, skip_prep=True),
                     ErrorDataset(paths[1], 131, skip_prep=True),
                     ErrorDataset(paths[2], 171, skip_prep=True),
                     ErrorDataset(paths[3], 193, skip_prep=True),
                     ErrorDataset(paths[4], 211, skip_prep=True),
                     ErrorDataset(paths[5], 335, skip_prep=True)
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
