from typing import Union

import numpy as np

import torch


class CropImage(object):
    """
    ***torchvision.Transforms compatible***

    Crops the height and width of the input image to the given size
    No point in really using this instead of torchvision
    """

    def __init__(self, size: Union[tuple, list] = (320, 320)):
        """
        Args:
            crop (tuple, list): The desired shape of the (H, W) of the output image
        """
        self.size = size

    def __call__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor (torch.Tensor): center crops the two last dims of the input
        Returns:
            torch.Tensor: Cropped torch.Tensor with the size of self.crop

        """
        shape = tensor.shape
        height, width = self.size

        img_h, img_w = shape[-2], shape[-1]
        if height > img_h:
            height = img_h
        if width > img_w:
            width = img_w

        return tensor[..., int(img_h/2 - height/2):int(img_h/2 + height/2),\
                      int(img_w/2 - width/2):int(img_w/2 + width/2)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
