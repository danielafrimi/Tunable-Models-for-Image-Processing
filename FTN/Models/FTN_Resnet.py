import torch.nn as nn
import torch
import torch.nn.functional as functional
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

    def __init__(self, in_channels, out_channels, alpha, kernel_size, stride=1, padding=None, bias=False,
                 ftn_layer=None):
        super(conv_ftn_block, self).__init__()
        if padding is None:
            if stride == 1:
                padding = (kernel_size - 1) // 2
            else:
                padding = 0
        # todo change to bias = True?
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.ftn = ftn_layer

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.check = []

    def forward(self, x):
        conv_layer_parameters = self.ftn(self.conv.weight)

        # Creating a convolution layer for operating the previous feature map
        # todo add cuda() on the layer while using GPU
        # generated_conv = get_conv_layer_with_updated_weights(conv_layer_parameters,
        #                                                      input_channels=64, output_channels=64)

        x = self.relu(self.bn(functional.conv2d(x, conv_layer_parameters, padding=1)))

        return x


class FTN_Resnet(nn.Module):
    def __init__(self, alpha, ftn_layers, input_channels=3):
        super(FTN_Resnet, self).__init__()

        self.ftn_layers = ftn_layers

        self.old_params = {}

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=64,
                                             kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        # todo change to bias True? change the numbers of layers when it fixed
        self.blocks = self._make_layers(conv_ftn_block, alpha, kernel_size=3, num_channels=64, num_of_layers=1,
                                        bias=False)

        self.output = nn.Conv2d(in_channels=64, out_channels=input_channels, kernel_size=3, stride=1, padding=1)

    def _make_layers(self, block, alpha, kernel_size, num_channels, num_of_layers, padding=1, bias=False):

        layers = [block(in_channels=num_channels, out_channels=num_channels, alpha=alpha, kernel_size=kernel_size,
                        padding=padding,
                        bias=bias, ftn_layer=self.ftn_layers[i]) for i in range(num_of_layers)]
        return nn.Sequential(*layers)

    def save_params(self):

        for name, params in self.named_parameters():
            self.old_params[name] = params.clone()

    def check_params(self):
        # perform update
        for name, params in self.named_parameters():
            if (self.old_params[name] == params).all():
                print("True")
                print(name)
            else:
                print("False")
                print(name)

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

    def get_ftn(self):
        return self.ftn_layers

    def __repr__(self):
        return 'FTN_RESNET'
