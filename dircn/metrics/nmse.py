import torch

class NMSE(torch.nn.Module):

    def __init__(self, mean: bool = True):
        self.mean = mean
        super().__init__()

    def forward(self, X, Y):
        """
        Args:
            X (torch.Tensor): prediction shape (batch, C, H, W)
            Y (torch.Tensor): ground truth shape (batch, C, H, W)
        returns:
            torch.Tensor: The NMSE for the input with shape (1, ) if mean else (batch, )
        """
        batch_size = X.shape[0]
        assert X.shape == Y.shape

        pred = X.view(batch_size, -1)
        gt = Y.view(batch_size, -1)
        if self.mean:
            return torch.mean(torch.norm(gt - pred, dim=1)**2/torch.norm(gt, dim=1)**2)
        else:
            return torch.norm(gt - pred, dim=1)**2/torch.norm(gt, dim=1)**2


if __name__=='__main__':
    a = torch.rand((15, 1, 150, 206))
    b = torch.rand((15, 1, 150, 206))
    met = NMSE()
    print(met(a, b))

    import numpy as np
    a = a.view(15, -1)
    b = b.view(15, -1)
    pred = a.numpy()
    gt = b.numpy()


    print(np.mean(np.linalg.norm(gt - pred, axis=1) ** 2 / np.linalg.norm(gt, axis=1) ** 2))
