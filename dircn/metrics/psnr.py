import torch


class PSNR(torch.nn.Module):

    def __init__(self, reduction: str = 'mean'):
        self.reduction = reduction
        super().__init__()


    def forward(self, X, Y):
        """
        Args:
            X (torch.Tensor): prediction shape (batch, C, H, W)
            Y (torch.Tensor): ground truth shape (batch, C, H, W)
        returns:
            torch.Tensor: The PSNR for the input with shape (1, ) if mean else (batch, )
        """
        batch_size = X.shape[0]
        assert X.shape == Y.shape

        pred = X.view(batch_size, -1)
        gt = Y.view(batch_size, -1)

        mse = torch.mean((pred - gt) ** 2, dim=1)

        # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        psnr = 20.0*torch.log10(pred.max(dim=1)[0]-pred.min(dim=1)[0]) - 10.0*torch.log10(mse)

        if self.reduction == 'mean':
            return torch.mean(psnr)
        elif self.reduction == 'sum':
            return torch.sum(psnr)
        else:
            return psnr

if __name__=='__main__':
    a = torch.rand((15, 1, 150, 206))
    b = torch.rand((15, 1, 150, 206))
    ps = PSNR()
    print(ps(a, b))
