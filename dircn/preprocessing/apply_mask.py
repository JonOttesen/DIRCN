from typing import Union, List

import numpy as np
import torch

class ApplyMaskColumn(object):
    """
    ***torchvision.Transforms compatible***

    Applies the given mask to the k-space data and sets the non
    specified columns to zero
    """

    def __init__(self, mask: Union[callable, List[callable]], random: bool = True):
        """
        Args:
            mask (torch.Tensor): The mask used in under-sampling the given k-space data,
                                 assumes shape: (number_of_columns_in_kspace)
        """
        self.mask = mask
        self.random = random
        self.index = 0

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor of the k-space data with
                                   shape (coil, rows, columns) or (rows, columns)
        Returns:
            torch.Tensor: K-space tensor with same shape and applied mask on columns

        """
        shape = tensor.shape + (1, )
        if isinstance(self.mask, list):
            if self.random:
                mask_generator = self.mask[np.random.randint(len(self.mask))]
            else:
                mask_generator = self.mask[self.index]
                self.index += 1
                self.index = self.index if self.index < len(self.mask) else 0
        else:
            mask_generator = self.mask

        mask, _ = mask_generator(shape)
        mask = mask.squeeze(0).squeeze(-1)

        print(mask.dtype, mask.shape)

        tensor[:, :, mask != True] = 0

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mask={0})'.format(self.mask)
