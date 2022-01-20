from typing import Union, List, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn

from .bottleneck import Bottleneck
from .basicblock import BasicBlock

class ResXUNet(nn.Module):

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        n: int = 64,
        groups: int = 32,
        bias: bool = False,
        ratio: float = 1./8,
        activation: Union[nn.Module, None] = None,
        interconnections: bool = False,
        make_interconnections: bool = False,
        ):
        super().__init__()
        self.interconnections = interconnections
        self.make_interconnections = make_interconnections

        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.activation = nn.ReLU(inplace=True) if activation is None else activation
        self.activation = nn.ReLU(inplace=True) if activation is None else activation


        self.input = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            nn.InstanceNorm2d(num_features=n, affine=bias),
            self.activation)

        self.inc = BasicBlock(
            channels=n,
            bias=bias,
            groups=groups,
            ratio=ratio,
            activation=self.activation,
            )


        self.down1 = DownConvBlock(
            in_channels=n,
            out_channels=2*n,
            groups=1,
            bias=bias,
            activation=self.activation,
            )

        self.down1_basic = BasicBlock(
            channels=2*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            activation=self.activation,
            )


        self.down2 = DownConvBlock(
            in_channels=2*n,
            out_channels=4*n,
            groups=1,
            bias=bias,
            activation=self.activation,
            )

        self.down2_basic = BasicBlock(
            channels=4*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            activation=self.activation,
            )


        self.down3 = DownConvBlock(
            in_channels=4*n,
            out_channels=8*n,
            groups=1,
            bias=bias,
            activation=self.activation,
            )

        self.down3_basic = BasicBlock(
            channels=8*n,
            bias=bias,
            groups=4*groups,
            ratio=ratio,
            activation=self.activation,
            )


        self.down4 = DownConvBlock(
            in_channels=8*n,
            out_channels=8*n,
            groups=1,
            bias=bias,
            activation=self.activation,
            )

        self.down4_bottle = nn.Sequential(*[Bottleneck(
            channels=8*n,
            mid_channels=8*n // 2,
            groups=4*groups,
            bias=bias,
            ratio=ratio,
            activation=self.activation,
            ) for i in range(2)])


        self.up4 = TransposeConvBlock(
            in_channels=8*n,
            out_channels=8*n,
            groups=1,
            bias=bias,
            activation=self.activation
            )




        self.up3_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=3*8*n if make_interconnections else 2*8*n,
                out_channels=8*n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            nn.InstanceNorm2d(num_features=8*n, affine=bias),
            self.activation)


        self.up3_basic = BasicBlock(
            channels=8*n,
            bias=bias,
            groups=4*groups,
            ratio=ratio,
            activation=self.activation,
            )

        self.up3 = TransposeConvBlock(
            in_channels=8*n,
            out_channels=4*n,
            groups=1,
            bias=bias,
            activation=self.activation
            )


        self.up2_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=3*4*n if make_interconnections else 2*4*n,
                out_channels=4*n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            nn.InstanceNorm2d(num_features=4*n, affine=bias),
            self.activation)


        self.up2_basic = BasicBlock(
            channels=4*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            activation=self.activation,
            )

        self.up2 = TransposeConvBlock(
            in_channels=4*n,
            out_channels=2*n,
            groups=1,
            bias=bias,
            activation=self.activation
            )




        self.up1_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=3*2*n if make_interconnections else 2*2*n,
                out_channels=2*n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            nn.InstanceNorm2d(num_features=2*n, affine=bias),
            self.activation)


        self.up1_basic = BasicBlock(
            channels=2*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            activation=self.activation,
            )

        self.up1 = TransposeConvBlock(
            in_channels=2*n,
            out_channels=n,
            groups=1,
            bias=bias,
            activation=self.activation
            )



        self.out_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=3*n if make_interconnections else 2*n,
                out_channels=n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            nn.InstanceNorm2d(num_features=n, affine=bias),
            self.activation)

        self.out_1 = BasicBlock(
            channels=n,
            bias=bias,
            groups=groups,
            ratio=ratio,
            activation=self.activation,
            )

        self.final_bottle = nn.Sequential(Bottleneck(
            channels=n,
            mid_channels=n,
            groups=groups,
            bias=bias,
            ratio=ratio,
            activation=self.activation,
            ))

        self.outc = nn.Conv2d(in_channels=n, out_channels=n_classes, stride=1, kernel_size=1)

    def forward(self, x: torch.Tensor, internals: Optional[List[torch.Tensor]] = None):

        x = self.input(x)
        x1 = self.inc(x)


        x2 = self.down1(x1)
        x2 = self.down1_basic(x2)

        x3 = self.down2(x2)
        x3 = self.down2_basic(x3)

        x4 = self.down3(x3)
        x4 = self.down3_basic(x4)

        x = self.down4(x4)
        x = self.down4_bottle(x)
        x = self.up4(x)

        if self.interconnections and internals is not None:
            assert len(internals) == 4, "When using dense cascading, all layers must be given"
            # Connect conv
            x1 = torch.cat([x1, internals[0]], dim=1)
            x2 = torch.cat([x2, internals[1]], dim=1)
            x3 = torch.cat([x3, internals[2]], dim=1)
            x4 = torch.cat([x4, internals[3]], dim=1)

        internals = list()

        x = torch.cat([x, x4], dim=1)
        x = self.up3_channel(x)
        x = self.up3_basic(x)
        internals.append(x)
        x = self.up3(x)

        x = torch.cat([x, x3], dim=1)
        x = self.up2_channel(x)
        x = self.up2_basic(x)
        internals.append(x)
        x = self.up2(x)

        x = torch.cat([x, x2], dim=1)
        x = self.up1_channel(x)
        x = self.up1_basic(x)
        internals.append(x)
        x = self.up1(x)

        x = torch.cat([x, x1], dim=1)
        x = self.out_channel(x)
        x = self.out_1(x)
        x = self.final_bottle(x)
        internals.append(x)

        internals.reverse()

        if self.interconnections:
            return self.outc(x), internals
        return self.outc(x)



# Additional blocks to make making the network easier (avoid unnecessary repeated code)

class DownConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 groups: int,
                 bias: bool,
                 activation: Union[nn.Module, None] = None,
                 ):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=2,
            stride=2,
            bias=False,
            padding=0,
            )

        self.norm = nn.InstanceNorm2d(num_features=out_channels, affine=bias)
        self.act = nn.ReLU(inplace=True) if activation is None else activation

    def forward(self, x: torch.Tensor):
        x = self.down(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class TransposeConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 groups: int,
                 bias: bool,
                 activation: Union[nn.Module, None] = None,
                 ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=2,
            stride=2,
            bias=False,
            padding=0,
            )
        self.norm = nn.InstanceNorm2d(num_features=out_channels, affine=bias)
        self.act = nn.ReLU(inplace=True) if activation is None else activation

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = self.norm(x)
        x = self.act(x)
        return x
