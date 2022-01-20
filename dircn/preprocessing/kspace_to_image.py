from packaging import version

import numpy as np
import torch

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore

import fMRI.fastmri as fastmri

import matplotlib.pyplot as plt

class KspaceToImage(object):
    """
    ***torchvision.Transforms compatible***

    Transforms the given k-space data to the corresponding image using Fourier transforms
    """
    def __init__(self,
                 norm: str = 'ortho',
                 shifting: bool = True,
                 complex_support: bool = True,
                 ):
        """
        Args:
            norm (str): normalization method used in the ifft transform,
                see doc for torch.fft.ifft for possible args
        """
        self.norm = norm
        self.shifting = shifting
        self.complex_support = complex_support

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor of the k-space data with shape
                                   (coils, rows, columns) i.e ndim=3
        Returns:
            torch.Tensor: The ifft transformed k-space to image with shape
                          (channels, rows, columns)
        """
        if isinstance(tensor, torch.Tensor):
            if self.shifting:
                if self.complex_support:
                    data = fastmri.ifftshift(tensor, dim=(-2, -1))
                    data = torch.fft.ifftn(data, dim=(-2, -1), norm=self.norm)
                    data = fastmri.fftshift(data, dim=(-2, -1))
                    return data
                else:
                    return fastmri.ifft2c(tensor)
            else:
                return torch.fft.ifftn(tensor, dim=(-2, -1), norm=self.norm)
        else:
            data = np.fft.ifftshift(tensor, axes=(-2, -1))
            data = np.fft.ifftn(data, axes=(-2, -1), norm=self.norm)
            data = np.fft.fftshift(data, axes=(-2, -1))
            return data

    def __repr__(self):
        return self.__class__.__name__ + '(norm={0}, shifting={1}, complex_support={2})'.format(self.norm, self.shifting, self.complex_support)
