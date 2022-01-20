"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        """
        Args:
            win_size (int, default=7): Window size for SSIM calculation.
            k1 (float, default=0.1): k1 parameter for SSIM calculation.
            k2 (float, default=0.03): k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range: float = None):

        if not data_range:
            """
            See for more info about data_range
            https://github.com/scikit-image/scikit-image/blob/master/skimage/metrics/_structural_similarity.py#L12-L232
            Assuming the first dimension is the batch_size
            """
            batch_size = Y.shape[0]

            Y_flattend = Y.view(batch_size, -1)
            data_range = Y_flattend.max(dim=1)[0] - Y_flattend.min(dim=1)[0]
            for i in range(len(Y.shape) - 1):
                data_range = torch.unsqueeze(data_range, dim=-1)

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()
