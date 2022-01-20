from dircn.masks.subsample import create_mask_for_mask_type
from fMRI.masks import KspaceMask

import torch

mask_genrator = create_mask_for_mask_type(
    "equispaced_fraction",
    center_fractions=[0.08, 0.04],
    accelerations=[4, 8],
    )

mask_4 = KspaceMask(acceleration=4, mask_type='equidistant', center_fraction=0.08, seed=None)  # 4x
mask, lines = mask_genrator((16, 320, 320, 2))
print(mask.shape)
print(mask_4(213).shape)
