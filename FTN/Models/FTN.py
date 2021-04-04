import torch.nn as nn
from collections import OrderedDict
import functools


class FTN(nn.Module):
    # FTN directly takes network kernels as inputs, instead of images or feature maps.

    def __init__(self, in_nc, out_nc, group_blocks):
        """
        todo input size of the FTN is quite small (usually 3 × 3 × C) because we operate it on the filter itself
        :param alpha:
        :param in_nc:
        :param out_nc:
        :param group_blocks: Number of blocked connections from input channels to output channels.
        Group convolution can reduce the number of parameters in a network. If the number of groups is increased,
        the degrees of freedom to change the original filters decrease.
        improve smoothness at the expense of adaptation and interpolation performance
        """
        super(FTN, self).__init__()

        # todo both kernels needs to be grouped
        self.conv1 = nn.Conv2d(in_nc, out_nc, kernel_size=(1, 1), groups=group_blocks)

        # PReLU(x)=max(0,x)+a∗min(0,x), init alpha with 1 (alpha is learnable)
        self.pReLu = nn.PReLU(init=1)
        # TODO needs identity init
        self.conv2 = nn.Conv2d(in_nc, out_nc, kernel_size=(1, 1), groups=group_blocks)

    def forward(self, x):
        """

        :param x: input - needs to be the filter itself
        :return:
        """

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

        self.FTN_layer = FTN(in_nc=in_nc, out_nc=out_nc, group_blocks=2)
        # todo check sizes
        self.conv = nn.Conv2d(out_nc, out_nc, kernel_size=(3, 3))

    def forward(self, x):
        print(x.shape)
        y = self.FTN_layer(x)
        print("y", y.shape)
        y = self.conv(y)
        y = y * self.alpha

        # todo the output is new filter and we need to conv with this?
        return (x * (1 - self.alpha)) + y


# train the main convolutional filter for the initial level with α = 0. Then, we freeze the main network and train
# the FTN only for the second level with α = 1, which breaks skip- connection. Next, the FTN learns the task
# transition itself. To that end, the FTN approximates kernels of the second level we can interpolate between two
# kernels (levels) by choosing α in the 0-1 range (in prediction time)
