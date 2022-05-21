import numpy as np
from astropy.visualization import ImageNormalize, LinearStretch
from iti.data.dataset import StackDataset, get_intersecting_files, BaseDataset
from iti.data.editor import BrightestPixelPatchEditor, LoadMapEditor, NormalizeRadiusEditor, AIAPrepEditor, \
    MapToDataEditor, NormalizeEditor, ReshapeEditor, LambdaEditor

# sdo_norms = {94: ImageNormalize(vmin=0, vmax=340, stretch=LinearStretch(), clip=False),
#              131: ImageNormalize(vmin=0, vmax=1400, stretch=LinearStretch(), clip=False),
#              171: ImageNormalize(vmin=0, vmax=8600, stretch=LinearStretch(), clip=False),
#              193: ImageNormalize(vmin=0, vmax=9800, stretch=LinearStretch(), clip=False),
#              211: ImageNormalize(vmin=0, vmax=5800, stretch=LinearStretch(), clip=False),
#              335: ImageNormalize(vmin=0, vmax=600, stretch=LinearStretch(), clip=False),
#              }
from sunpy.map import Map

sdo_norms = {94: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             131: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             171: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             193: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             211: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             335: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             }

class LinearAIADataset(BaseDataset):

    def __init__(self, data, wavelength, resolution=4096, ext='.fits', **kwargs):
        norm = sdo_norms[wavelength]

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   AIAPrepEditor(),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution)),
                   LambdaEditor(lambda d: np.clip(d, a_min=-1, a_max=None, dtype=np.float32))]
        super().__init__(data, editors=editors, ext=ext, **kwargs)


class DEMDataset(StackDataset):

    def __init__(self, data, patch_shape=None, resolution=4096, ext='.fits', **kwargs):
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, ['94', '131', '171', '193', '211', '335'], ext=ext, **kwargs)
        data_sets = [LinearAIADataset(paths[0], 94, resolution=resolution),
                     LinearAIADataset(paths[1], 131, resolution=resolution),
                     LinearAIADataset(paths[2], 171, resolution=resolution),
                     LinearAIADataset(paths[3], 193, resolution=resolution),
                     LinearAIADataset(paths[4], 211, resolution=resolution),
                     LinearAIADataset(paths[5], 335, resolution=resolution)
                     ]
        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape, random_selection=0))

def prep_map(s_map, resolution=4096):
    norm = sdo_norms[int(s_map.wavelength.value)]
    radius_editor = NormalizeRadiusEditor(resolution)
    aia_prep_editor = AIAPrepEditor()

    s_map = radius_editor.call(s_map)
    s_map = aia_prep_editor.call(s_map)
    data = norm(s_map.data).astype(np.float32) * 2 - 1
    data = np.clip(data, a_min=-1, a_max=None, dtype=np.float32)
    s_map = Map(data, s_map.meta)
    return s_map