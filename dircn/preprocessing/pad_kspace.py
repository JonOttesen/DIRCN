from typing import Union, Tuple, List

import numpy as np

import torch


class PadKspace(object):
    """
    ***torchvision.Transforms compatible***

    Pads kspace to the given shape in both directions
    """
    def __init__(self, shape: Union[int, Tuple[int], List[int]] = (320, 320)):
        shape = (shape, shape) if isinstance(shape, int) else shape
        self.col, self.row = shape

    def __call__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor (torch.Tensor, np.ndarray): The input tensor/array, for tensor assumes shape (coil, h, w)
        Returns:
            (torch.Tensor, np.ndarray): The padded version of the input
        """
        shape = tensor.shape

        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
            is_tensor = True
        else:
            is_tensor = False

        padding = np.array([[0, 0],
                            [np.ceil((self.col - shape[1])/2), np.floor((self.col - shape[1])/2)],
                            [np.ceil((self.row - shape[2])/2), np.floor((self.row - shape[2])/2)]]).astype(np.int32)
        padding[padding < 0] = 0

        tensor = np.pad(
            array=tensor,
            pad_width=padding,
            mode='constant',
            constant_values=0,
            )

        return tensor if is_tensor is False else torch.from_numpy(tensor)

