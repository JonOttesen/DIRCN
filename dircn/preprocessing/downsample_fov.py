from typing import Union

import numpy as np
import torch

import torchvision

from .kspace_to_image import KspaceToImage
from .image_to_kspace import ImageToKspace
from .crop_image import CropImage

import dircn.fastmri as fastmri


class DownsampleFOV(object):
    """
    ***torchvision.Transforms compatible***

    Downsamples the FOV by fourier than cropping than inverse fouirer
    """
    def __init__(self, k_size: int = 320, i_size: int = 320, complex_support: bool = True, quadratic: bool = False):
        """
        Args:
            dim (int): the dimension for downsampling, 1 for height and 2 for width
            size (int): the length of k-space along the dim direction
        """
        self.k_size = k_size
        self.i_size = i_size
        self.complex_support = complex_support
        self.quadratic = quadratic

    def _numpy(self, tensor: np.ndarray):
        fft = KspaceToImage(norm='ortho')
        ifft = ImageToKspace(norm='ortho')
        i_crop = CropImage((self.i_size, self.i_size))
        k_crop = CropImage((self.k_size, self.k_size))

        tensor = fft(tensor)
        tensor = i_crop(tensor)
        tensor = ifft(tensor)
        if not self.i_size == self.k_size:
            tensor = k_crop(tensor)

        return tensor

    def __call__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Calculates the phase images
        Args:
            tensor (torch.Tensor): Complex Tensor of the image data with shape
                                   (coils, rows, columns, ...) i.e ndim >= 1
        Returns:
            torch.Tensor: The real phase images with equal shape

        """
        if self.quadratic:
            self.i_size = min(tensor.shape[-1], tensor.shape[-2])
            self.k_size = self.i_size
        if isinstance(tensor, np.ndarray):
            return self._numpy(tensor)

        if not self.complex_support:
            fft = fastmri.fft2c
            ifft = fastmri.ifft2c
            i_crop = CropImage((self.i_size, self.i_size))
            k_crop = CropImage((self.k_size, self.k_size))
        else:
            fft = KspaceToImage(norm='ortho')
            ifft = ImageToKspace(norm='ortho')
            i_crop = CropImage((self.i_size, self.i_size))
            k_crop = CropImage((self.k_size, self.k_size))


        tensor = fft(tensor)
        if not self.complex_support:
            order = list(range(len(tensor.shape)))
            order[-3] += 2
            order[-1] -= 2
            tensor = tensor.permute(order)

        tensor = i_crop(tensor)

        if not self.complex_support:
            tensor = tensor.permute(order)

        tensor = ifft(tensor)

        if self.k_size != self.i_size:
            if not self.complex_support:
                tensor = tensor.permute(order)
            tensor = k_crop(tensor)

            if not self.complex_support:
                tensor = tensor.permute(order)

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(k_size={0}, i_size={1}, complex_support={2})'.format(self.k_size, self.i_size, self.complex_support)
