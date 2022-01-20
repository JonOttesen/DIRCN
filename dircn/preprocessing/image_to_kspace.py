from packaging import version

import numpy as np
import torch
if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore

import fMRI.fastmri as fastmri
import matplotlib.pyplot as plt

class ImageToKspace(object):
    """
    ***torchvision.Transforms compatible***

    Transforms the given image data to the corresponding image using Fourier transforms
    """
    def __init__(self,
                 norm: str = 'ortho',
                 ):
        """
        Args:
            norm (str): normalization method used in the fft transform,
                see doc for torch.fft.fft for possible args
        """
        self.norm = norm

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor of the image data with shape
                                   (coils, rows, columns) i.e ndim=3
        Returns:
            torch.Tensor: The fft transformed image to k-space with shape
                          (channels, rows, columns)

        """
        if isinstance(tensor, torch.Tensor):
            tensor_dtype = tensor.dtype
            tensor = tensor.type(torch.complex128)

            data = fastmri.fftshift(tensor, dim=(-2, -1))
            data = torch.fft.fftn(data, dim=(-2, -1), norm=self.norm)
            data = fastmri.ifftshift(data, dim=(-2, -1))
            return data.type(tensor_dtype)
        else:
            numpy_dtype = tensor.dtype
            tensor = tensor.astype(np.complex128)
            data = np.fft.fftshift(tensor, axes=(-2, -1))
            data = np.fft.fftn(data, axes=(-2, -1), norm=self.norm)
            data = np.fft.ifftshift(data, axes=(-2, -1))
            return data.astype(numpy_dtype)

    def __repr__(self):
        return self.__class__.__name__ + '(norm={0})'.format(self.norm)
