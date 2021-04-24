import torch.nn as nn


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

        # self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
    #             clip_b = 0.025
    #             w = m.weight.data.shape[0]
    #             for j in range(w):
    #                 if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
    #                     m.weight.data[j] = clip_b
    #                 elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
    #                     m.weight.data[j] = -clip_b
    #             m.running_var.fill_(0.01)

    def forward(self, x):
        # x is the filter weights
        identity = x
        x = self.conv1(x)
        x = self.pReLu(x)
        x = self.conv2(x)

        return x + identity


class FTNBlock(nn.Module):
    def __init__(self, alpha: float, in_nc, out_nc):
        """
        :param alpha: Value between [0,1] that control the weighted sum of the filters
        :param in_nc: Number of incoming channels
        :param out_nc: Number of outgoing channels
        """

        super(FTNBlock, self).__init__()
        self.alpha = alpha

        # The FTN layer get kernel as input and produce a tensor of the same size
        self.FTN_layer = FTN(in_nc=in_nc, out_nc=in_nc, group_blocks=64)
        self.conv = nn.Conv2d(in_nc, in_nc, kernel_size=(1, 1)).requires_grad_(True)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.conv.weight)

    def forward(self, x):
        input_filter = x
        y = self.FTN_layer(x)
        y = self.conv(y)

        # Weighted sum of both filters
        return (input_filter * (1 - self.alpha)) + (y * self.alpha)

# train the main convolutional filter for the initial level with α = 0. Then, we freeze the main network and train
# the FTN only for the second level with α = 1, which breaks skip- connection. Next, the FTN learns the task
# transition itself. To that end, the FTN approximates kernels of the second level we can interpolate between two
# kernels (levels) by choosing α in the 0-1 range (in prediction time)
