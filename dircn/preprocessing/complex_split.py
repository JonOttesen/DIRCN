import torch

from dircn.fastmri.coil_combine import rss

class ComplexSplit(object):
    """
    ***torchvision.Transforms compatible***

    Splits complex tensor by adding anoter dim
    input (x, y, z) -> (x, y, z, 2) where [0] is the real part and [1] is the imag
    """

    def __call__(self, tensor: torch.Tensor):
        """
        Splits a complex tensor by adding a new dim
        Args:
            tensor (torch.Tensor): tensor with some shape (x, y, ..., z)
        Returns:
            torch.Tensor: the splitted torch tensor (x, y, ..., z, 2)

        """
        return torch.view_as_real(tensor)


    def __repr__(self):
        return self.__class__.__name__
