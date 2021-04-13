import torch.nn as nn
import torch
from Models import FTN

def get_conv_layer_with_updated_weights(conv_layer_parameters, input_channels, output_channels):
    # We dont want to learn this layer, we only use it on the input feature map
    generated_conv = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1, padding_mode='zeros') \
        .requires_grad_(False)
    # Load parameters into the layer
    generated_conv.weight = nn.Parameter(conv_layer_parameters)
    # todo what do we need to do with the bias?
    # generated_conv1.bias = nn.Parameter(initial_param['bias'])
    return generated_conv


class conv_ftn_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, bias=False):
        super(conv_ftn_block, self).__init__()
        if padding == None:
            if stride == 1:
                padding = (kernel_size - 1) // 2
            else:
                padding = 0
        # todo change to bias = True?
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.ftn = FTN.FTNBlock(alpha=0, in_nc=64, out_nc=64)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_layer_parameters = self.ftn(self.conv.weight)

        # Creating a convolution layer for operating the previous feature map
        generated_conv = get_conv_layer_with_updated_weights(conv_layer_parameters,
                                                             input_channels=64, output_channels=64)
        x = self.relu(self.bn(generated_conv(x)))

        return x


class FTN_Resnet(nn.Module):
    def __init__(self, input_channels=3):
        super(FTN_Resnet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=64,
                                             kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        # todo change to bias True?
        self.blocks = self._make_layers(conv_ftn_block, kernel_size=3, num_channels=64, num_of_layers=10, bias=False)

        self.output = nn.Conv2d(in_channels=64, out_channels=input_channels, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block, kernel_size, num_channels, num_of_layers, padding=1, bias=False):

        layers = [block(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=padding,
                         bias=bias) for _ in range(num_of_layers)]
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.blocks(x)
        return self.output(x) + residual

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])

    def __repr__(self):
        return 'FTN_RESNET'
