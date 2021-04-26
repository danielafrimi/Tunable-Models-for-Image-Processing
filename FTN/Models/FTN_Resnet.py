import torch.nn as nn
import torch
import torch.nn.functional as functional
from Models import FTN
from torch.nn.parameter import Parameter


class Kernels(nn.Module):

    def __init__(self, z_num=(3, 3), z_dim=64):
        super(Kernels, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim
        self.kernel_weights_arr = []

        self.z_list.append(Parameter(torch.fmod(torch.randn(64, 64, 3, 3), 2)))

    def forward(self, hyper_net):
        kernel_weights = hyper_net(self.z_list[0])
        self.kernel_weights_arr.append(kernel_weights)
        return kernel_weights


class conv_ftn_block(nn.Module):

    def __init__(self, in_channels, out_channels, alpha, kernel_size, stride=1, padding=None, bias=False,
                 ftn_layer=None):
        super(conv_ftn_block, self).__init__()
        if padding is None:
            if stride == 1:
                self.padding = (kernel_size - 1) // 2
            else:
                self.padding = 0
        else:
            self.padding = 1

        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        #                       kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, conv_layer_parameters):
        residual = x

        # Creating a convolution layer for operating the previous feature map
        x = self.relu(self.bn(functional.conv2d(x, conv_layer_parameters, padding=self.padding )))

        return x + residual


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

        self.zs = nn.ModuleList()
        self.zs.append(Kernels())

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
                print("True", name)
            else:
                print("False", name)

    def forward(self, x):
        x = self.conv1(x)

        conv_weights = self.zs[0](self.ftn_layers[0])

        x = self.blocks[0](x, conv_weights)
        return self.output(x)

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])

    def get_ftn(self):
        return self.ftn_layers

    def __repr__(self):
        return 'FTN_RESNET'
