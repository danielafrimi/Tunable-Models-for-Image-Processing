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

    def forward(self, x):
        # x is the filter weights - torch.Size([64, 64, 3, 3])
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
        # self.FTN_layer = FTN(in_nc=in_nc, out_nc=in_nc, group_blocks=64) # todo delete?

        # input size of the FTN is quite small (usually 3 × 3 × C) because we operate it on the filter itself
        self.conv1_ftn = nn.Conv2d(in_nc, out_nc, kernel_size=(1, 1), groups=16).requires_grad_(True)

        # PReLU(x)=max(0,x)+a∗min(0,x), init alpha with 1 (alpha is learnable)
        self.pReLu = nn.PReLU(init=1).requires_grad_(True)

        # todo maybe change the out_nc and in_nc to be 16? for the group as they said in the paper???
        self.conv2_ftn = nn.Conv2d(out_nc, in_nc, kernel_size=(1, 1), groups=16).requires_grad_(True)
        self.conv3_ftn = nn.Conv2d(in_nc, in_nc, kernel_size=(1, 1)).requires_grad_(True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

        # todo initialize with identity filter and zero biases????
        # weights = torch.tensor([[0., 0., 0.],
        #                         [0., 1., 0.],
        #                         [0., 0., 0.]])
        # weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
        # output = F.conv2d(x, weights) todo or define a conv object and than
        # with torch.no_grad():
        #     conv.weight = nn.Parameter(weights)

    def forward(self, x):
        input_filter = x
        # x is the filter weights - torch.Size([64, 64, 3, 3])

        # FTN Layer
        x = self.conv1_ftn(x)
        x = self.pReLu(x)
        x = self.conv2_ftn(x) + input_filter

        y = self.conv3_ftn(x)

        # Weighted sum of both filters
        return (input_filter * (1 - self.alpha)) + (y * self.alpha)

# train the main convolutional filter for the initial level with α = 0. Then, we freeze the main network and train
# the FTN only for the second level with α = 1, which breaks skip- connection. Next, the FTN learns the task
# transition itself. To that end, the FTN approximates kernels of the second level we can interpolate between two
# kernels (levels) by choosing α in the 0-1 range (in prediction time)
