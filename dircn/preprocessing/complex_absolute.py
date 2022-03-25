from typing import Union

import torch
import numpy as np

import dircn.fastmri as fastmri

class ComplexAbsolute(object):
    """
    ***torchvision.Transforms compatible***

    Calculates the Complex absolute i.e sqrt(x**2 + y**2) between the real
    and imaginary parts of the image which is the last dimension
    """

    def __call__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Takes the complex absolute of the last dimension
        Args:
            tensor (torch.Tensor): Tensor of the complex image (but doesn't need to).
                                   (coils/channels, rows, columns, ...., complex) ndim >= 1
        Returns:
            torch.Tensor: The ifft transformed k-space to image with shape
                          (channels, rows, columns) or
                          (channels, rows, columns, complex)
                          if the complex_absolute is not calculated

        """
        if isinstance(tensor, np.ndarray):
            return np.absolute(tensor)

        if tensor.size(-1) == 2:
            return fastmri.math.complex_abs(data=tensor)
        else:
            return tensor.abs()

    def __repr__(self):
        return self.__class__.__name__
