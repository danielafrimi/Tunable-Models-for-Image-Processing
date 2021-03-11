import torch.nn as nn

class FTNLayer(nn.Module):
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
        super(FTNLayer, self).__init__()
        # TODO needs identity init
        self.conv1 = nn.Conv2d(in_nc, out_nc, kernel_size=1, groups=group_blocks)
        # PReLU(x)=max(0,x)+a∗min(0,x), init alp[ha with 1 (alpha is learnable)
        self.pReLu = nn.PReLU(init=1)
        # TODO needs identity init
        self.conv2 = nn.Conv2d(in_nc, out_nc, kernel_size=1)

    def forward(self, x):
        # x is the filter - todo check the shapes of this
        x = self.conv1(x)
        x = self.pReLu(x)
        x = self.conv2(x)
        return x

class FTNBlock(nn.Module):
    def __init__(self, alpha: float, in_nc, out_nc):

        super(FTNBlock, self).__init__()
        self.alpha = alpha
        self.conv1 = nn.Conv2d(in_nc, out_nc, kernel_size=3)
        self.FTN_layer = FTNLayer(in_nc=self.conv1.shape[0], out_nc=self.conv1.shape[1], group_blocks=4)
        # todo check sizes
        self.conv2 = nn.Conv2d(out_nc, out_nc, kernel_size=3)



    def forward(self, x):
        x = self.conv1(x)
        # todo check this
        y = self.FTN_layer.forward(x)
        y = self.conv2(y)
        y = y * self.alpha
        # the output is new filter and we need to conv with this ?
        return (x * (1 - self.alpha)) + y



# train the main convolutional filter for the initial level with α = 0.
# Then, we freeze the main network and train the FTN only for the second level with α = 1, which breaks skip- connection.
# Next, the FTN learns the task transition itself. To that end, the FTN approximates kernels of the second level
# we can interpolate between two kernels (levels) by choosing α in the 0-1 range (in prediction time)