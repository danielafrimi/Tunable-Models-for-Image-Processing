import torch
import torch.nn as nn


# train the main convolutional filter for the initial level with α = 0. Then, we freeze the main network and train
# the FTN only for the second level with α = 1, which breaks skip- connection. Next, the FTN learns the task
# transition itself. To that end, the FTN approximates kernels of the second level we can interpolate between two
# kernels (levels) by choosing α in the 0-1 range (in prediction time)


class FTN(nn.Module):

    def __init__(self, in_nc, out_nc, group_blocks):
        """
        FTN directly takes network kernels as inputs, instead of images or feature maps.
        :param in_nc: Number of incoming channels
        :param out_nc:  Number of outgoing channels
        :param group_blocks: Number of blocked connections from input channels to output channels.
        Group convolution can reduce the number of parameters in a network. If the number of groups is increased,
        the degrees of freedom to change the original filters decrease.
        improve smoothness at the expense of adaptation and interpolation performance
        """
        super(FTN, self).__init__()

        # input size of the FTN is quite small (usually 3 × 3 × C) because we operate it on the filter itself
        self.conv1 = nn.Conv2d(in_nc, out_nc, kernel_size=(1, 1), groups=group_blocks).requires_grad_(True)

        # PReLU(x)=max(0,x)+a∗min(0,x), init alpha with 1 (alpha is learnable)
        self.pReLu = nn.PReLU(init=1)
        self.conv2 = nn.Conv2d(out_nc, in_nc, kernel_size=(1, 1), groups=group_blocks).requires_grad_(True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.pReLu(x)
        x = self.conv2(x)

        return x + identity


class FTNBlock(nn.Module):
    def __init__(self, alpha: float, in_nc, out_nc, groups):
        """
        :param alpha: Value between [0,1] that control the weighted sum of the filters
        :param in_nc: Number of incoming channels
        :param out_nc: Number of outgoing channels
        """

        super(FTNBlock, self).__init__()
        self.alpha = alpha

        # The FTN layer get kernel as input and produce a tensor of the same size
        # self.FTN_layer = FTN(in_nc=in_nc, out_nc=in_nc, group_blocks=64) # todo delete?

        # input size of the FTN is quite small (usually 3 × 3 × C) because we operate it on the filter itself
        self.conv1_ftn = nn.Conv2d(in_nc, out_nc, kernel_size=(1, 1), groups=groups).requires_grad_(True)

        # PReLU(x)=max(0,x)+a∗min(0,x), init alpha with 1 (alpha is learnable)
        self.pReLu = nn.PReLU(init=1).requires_grad_(True)

        # todo maybe change the out_nc and in_nc to be 16 - doenst work? for the group as they said in the paper???
        # todo maybe to change it back to 64 for learning filter for each feature map
        self.conv2_ftn = nn.Conv2d(out_nc, in_nc, kernel_size=(1, 1), groups=groups).requires_grad_(True)
        self.conv3_ftn = nn.Conv2d(in_nc, in_nc, kernel_size=(1, 1)).requires_grad_(True)

    def init_weights(self, identity=True):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if identity:
                    weights = torch.Tensor([[1]])
                    weights = weights.view(1, 1, 1, 1).repeat(m.weight.shape[0], m.weight.shape[1], 1, 1)
                    bias_weights = torch.zeros(m.bias.shape)

                    with torch.no_grad():
                        m.weight = nn.Parameter(weights)
                        m.bias = nn.Parameter(bias_weights)
                else:
                    m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

    def forward(self, x):
        # todo play with the bias? with the interpolation???????????
        input_filter = x

        # FTN Layer
        x = self.conv1_ftn(x)
        x = self.pReLu(x)
        x = self.conv2_ftn(x) + input_filter

        y = self.conv3_ftn(x)

        # Weighted sum of both filters
        # todo maybe return tuple with the bias parameters
        return (input_filter * (1 - self.alpha)) + (y * self.alpha)
