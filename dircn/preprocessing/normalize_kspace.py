from typing import Union

import torch
import numpy as np

import dircn.fastmri as fastmri

class NormalizeKspace(object):
    """
    ***torchvision.Transforms compatible***

    Downsamples the FOV by fourier than cropping than inverse fouirer
    """
    def __init__(self,
                 center_fraction: float = 0.08,
                 return_max: bool = False,
                 complex_support: bool = True,
                 multiply: float = 1.0,
                 ):
        """
        Args:
            dim (int): the dimension for downsampling, 1 for height and 2 for width
            size (int): the length of k-space along the dim direction
        """
        self.center_fraction = center_fraction
        self.return_max = return_max
        self.complex_support = complex_support
        self.multiply = multiply

    def __call__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Calculates the phase images
        Args:
            tensor (torch.Tensor): Complex Tensor of the image data with shape
                                   (coils, rows, columns, ...) i.e ndim >= 1
        Returns:
            torch.Tensor: The real phase images with equal shape

        """
        lines = tensor.shape[2]
        cent = tensor.shape[2] // 2
        frac = int(lines*self.center_fraction // 2)

        if isinstance(tensor, np.ndarray):
            x = tensor.copy()
            mxx = np.absolute(x[:, :, cent-frac:cent+frac]).max()
        else:
            x = tensor.clone()
            if self.complex_support:
                mxx = torch.max(torch.abs(x[:, :, cent-frac:cent+frac]))
            else:
                mxx = torch.max(fastmri.math.complex_abs(x[:, :, cent-frac:cent+frac]))

        tensor = tensor/mxx
        tensor = tensor*self.multiply

        if self.return_max:
            return tensor, mxx/self.multiply

        return tensor


    def __repr__(self):
        return self.__class__.__name__ + '(center_fraction={0}, return_max={1}, complex_support={2})'.format(self.center_fraction, self.return_max, self.complex_support)

