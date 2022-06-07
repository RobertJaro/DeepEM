import numpy as np
from astropy.visualization import ImageNormalize, LinearStretch
from iti.data.dataset import StackDataset, get_intersecting_files, BaseDataset
from iti.data.editor import BrightestPixelPatchEditor, LoadMapEditor, AIAPrepEditor, \
    MapToDataEditor, NormalizeEditor, LambdaEditor, ExpandDimsEditor
from sunpy.map import Map

sdo_norms = {94: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             131: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             171: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             193: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             211: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             335: ImageNormalize(vmin=0, vmax=1e4, stretch=LinearStretch(), clip=False),
             }


class LinearAIADataset(BaseDataset):

    def __init__(self, data, wavelength, ext='.fits', **kwargs):
        norm = sdo_norms[wavelength]

        editors = [LoadMapEditor(),
                   AIAPrepEditor('aiapy'),
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


def prep_map(s_map):
    norm = sdo_norms[int(s_map.wavelength.value)]
    aia_prep_editor = AIAPrepEditor()

    s_map = aia_prep_editor.call(s_map)
    data = norm(s_map.data).astype(np.float32) * 2 - 1
    data = np.clip(data, a_min=-1, a_max=10, dtype=np.float32)
    s_map = Map(data, s_map.meta)
    return s_map
