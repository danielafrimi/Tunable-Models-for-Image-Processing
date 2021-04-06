import torch.nn as nn
from collections import OrderedDict
import functools


class FTN(nn.Module):
    # FTN directly takes network kernels as inputs, instead of images or feature maps.

    def __init__(self, in_nc, out_nc, group_blocks):
        """

        :param alpha:
        :param in_nc:
        :param out_nc:
        :param group_blocks: Number of blocked connections from input channels to output channels.
        Group convolution can reduce the number of parameters in a network. If the number of groups is increased,
        the degrees of freedom to change the original filters decrease.
        improve smoothness at the expense of adaptation and interpolation performance
        """
        super(FTN, self).__init__()

        # input size of the FTN is quite small (usually 3 × 3 × C) because we operate it on the filter itself
        self.conv1 = nn.Conv2d(in_nc, out_nc, kernel_size=(1, 1), groups=group_blocks)

        # PReLU(x)=max(0,x)+a∗min(0,x), init alpha with 1 (alpha is learnable)
        self.pReLu = nn.PReLU(init=1)
        self.conv2 = nn.Conv2d(out_nc, in_nc, kernel_size=(1, 1), groups=group_blocks)

    def forward(self, x):
        # x is the filter - the shape is torch.Size([64, 3, 3, 3])
        identity = x
        x = self.conv1(x)
        x = self.pReLu(x)
        x = self.conv2(x)
        # todo needs to be skip connection with weighted sum - what does it mean with weights sum?

        return x + identity


class FTNBlock(nn.Module):
    def __init__(self, alpha: float, in_nc, out_nc):

        super(FTNBlock, self).__init__()
        self.alpha = alpha
        self.input_channels = in_nc
        self.output_channels = out_nc

        # The FTN layer get kernel as input and produce a tensor of the same size
        self.FTN_layer = FTN(in_nc=in_nc, out_nc=in_nc, group_blocks=3)
        # todo check sizes
        self.conv = nn.Conv2d(in_nc, in_nc, kernel_size=(1, 1))

    def forward(self, x):
        # x shape is torch.Size([60, 3, 3, 3]) -> 60 channels as output, 3 channels as input, and filter size of 3,3
        input_filter = x
        y = self.FTN_layer(x)
        # y shape is torch.Size([60, 3, 3, 3]) same as x
        y = self.conv(y)

        return (input_filter * (1 - self.alpha)) + (y * self.alpha)


# train the main convolutional filter for the initial level with α = 0. Then, we freeze the main network and train
# the FTN only for the second level with α = 1, which breaks skip- connection. Next, the FTN learns the task
# transition itself. To that end, the FTN approximates kernels of the second level we can interpolate between two
# kernels (levels) by choosing α in the 0-1 range (in prediction time)

