from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_window(size: int = 11,
                    sigma: float = 1.5,
                    channels: int = 1,
                    gaussian: bool = True):
    if gaussian:
        x = torch.arange(start=0, end=size, step=1) - size // 2
        gauss = torch.exp(-x**2/(2*sigma**2)).unsqueeze(0)
        gauss = (gauss.T @ gauss).unsqueeze(0)
        gauss /= gauss.sum()
        gauss = gauss.unsqueeze(0)
        gauss = torch.cat([gauss]*channels, dim=0)
        return gauss

    return torch.ones((size, size)).unsqueeze(0).unsqueeze(0)/size**2

class SSIM(nn.Module):
    """
    Original Paper: http://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
    """

    def __init__(self,
                 size: int = 7,
                 sigma: float = 1.5,  # 1.5 is the regular for a size of 11
                 channels: int = 1,
                 k1: float = 0.01,
                 k2: float = 0.03):

        super(SSIM, self).__init__()
        self.size = size
        self.sigma = sigma
        self.channels = channels
        self.k1 = k1
        self.k2 = k2
        # Original approach, gives grad = None so no problemo, but the other approach is better
        # self.gaussian_window = nn.Parameter(gaussian_window(size=self.size,
                                                                # sigma=self.sigma,
                                                                # channels=channels,
                                                                # ))
        self.register_buffer('gaussian_window', gaussian_window(size=self.size,
                                                                sigma=self.sigma,
                                                                channels=channels,
                                                                ))

        NP = size ** 2
        self.cov_norm = NP / (NP - 1)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, sigma={1}, channels={2}, k1={3}, k2={4})'.format(
            self.size, self.sigma, self.channels, self.k1, self.k2)

    def apply_conv2d(self, X: torch.Tensor, pad: int = 0):
        channels = X.shape[1]
        return F.conv2d(input=X,
                        weight=self.gaussian_window,
                        stride=1,
                        padding=pad,
                        groups=channels)

    def forward(self,
                X: torch.Tensor,
                Y: torch.Tensor,
                data_range: float = None):
        """
        Args:
            X (torch.Tensor): Prediction
            Y (torch.Tensor): Ground truth
            data_range (float): Difference between maximum and minimum value
        """
        if not data_range:
            """
            See for more info about data_range
            https://github.com/scikit-image/scikit-image/blob/master/skimage/metrics/_structural_similarity.py#L12-L232
            Assuming the first dimension is the batch_size
            """
            # Original implementation
            # data_range = 2  # Since the std=1 i.e 1 -- 1 = 2

            # This is originally X not Y, but I will try using the Y instead
            batch_size = Y.shape[0]
            Y_flattend = Y.view(batch_size, -1)
            data_range = Y_flattend.max(dim=1)[0] - Y_flattend.min(dim=1)[0]
            for i in range(len(Y.shape) - 1):
                data_range = torch.unsqueeze(data_range, dim=-1)

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        pad = self.size // 2

        mux = self.apply_conv2d(X, pad=pad)
        muy = self.apply_conv2d(Y, pad=pad)

        mux_sq = mux.pow(2)
        muy_sq = muy.pow(2)
        muxy = mux * muy

        sigmax_sq = (self.apply_conv2d(X * X, pad=pad) - mux_sq)*self.cov_norm
        sigmay_sq = (self.apply_conv2d(Y * Y, pad=pad) - muy_sq)*self.cov_norm
        sigmaxy = (self.apply_conv2d(X * Y, pad=pad) - muxy)*self.cov_norm

        ssim_map = ((2*muxy + C1)*(2*sigmaxy + C2)/
                   ((mux_sq + muy_sq + C1)*(sigmax_sq + sigmay_sq + C2)))

        return 1 - ssim_map.mean()


class MS_SSIM(object):
    """
    Supposedly more robust version of the SSIM
    Code Source: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    Possible later implementation
    """



if __name__=='__main__':
    torch.manual_seed(42)
    a = torch.rand((15, 1, 150, 206))
    b = torch.rand((15, 1, 150, 206))


    loss = SSIM(channels=1)
    g = loss(a, b)
    for p in loss.parameters():
        print(p.grad)
