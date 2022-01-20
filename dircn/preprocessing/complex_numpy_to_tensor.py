import torch
import numpy as np

class ComplexNumpyToTensor(object):
    """
    ***torchvision.Transforms compatible***

    Converts a numpy complex array to a torch tensor where the real and
    imaginary parts are stacked along the last dimension
    """
    def __init__(self, complex_support: bool = True):
        self.complex_support = complex_support

    def __call__(self, tensor: np.ndarray):
        """
        Args:
            tensor (np.ndarry): Array with shape (batch, coils, rows, columns)
        Returns:
            torch.Tensor: The torch.Tensor version of the complex numpy array
                          with shape (batch, coils, rows, columns) with the
                          last dim being the real and complex part

        """
        if self.complex_support:
            return torch.from_numpy(tensor)
        else:
            data = torch.zeros(tensor.shape + (2, ))
            data[:, ..., 0] = torch.from_numpy(tensor.real)
            data[:, ..., 1] = torch.from_numpy(tensor.imag)
            return data


    def __repr__(self):
        return self.__class__.__name__ + '(complex_support={0})'.format(self.complex_support)
