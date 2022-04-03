from astropy.visualization import ImageNormalize, LinearStretch
from iti.data.dataset import StackDataset, get_intersecting_files, BaseDataset
from iti.data.editor import BrightestPixelPatchEditor, LoadMapEditor, NormalizeRadiusEditor, AIAPrepEditor, \
    MapToDataEditor, NormalizeEditor, ReshapeEditor

sdo_norms = {94: ImageNormalize(vmin=0, vmax=340, stretch=LinearStretch(), clip=True),
             131: ImageNormalize(vmin=0, vmax=1400, stretch=LinearStretch(), clip=True),
             171: ImageNormalize(vmin=0, vmax=8600, stretch=LinearStretch(), clip=True),
             193: ImageNormalize(vmin=0, vmax=9800, stretch=LinearStretch(), clip=True),
             211: ImageNormalize(vmin=0, vmax=5800, stretch=LinearStretch(), clip=True),
             335: ImageNormalize(vmin=0, vmax=600, stretch=LinearStretch(), clip=True),
             }


class LinearAIADataset(BaseDataset):

    def __init__(self, data, wavelength, resolution=2048, ext='.fits', **kwargs):
        norm = sdo_norms[wavelength]

        editors = [LoadMapEditor(),
                   NormalizeRadiusEditor(resolution),
                   AIAPrepEditor(),
                   MapToDataEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((1, resolution, resolution))]
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
            self.addEditor(BrightestPixelPatchEditor(patch_shape))
