import torch.nn as nn
import torch
import torch.nn.functional as functional
from Models import FTN
from torch.nn.parameter import Parameter


class Kernels(nn.Module):

    def __init__(self, z_num=(3, 3), z_dim=64):
        super(Kernels, self).__init__()

        self.z_list = nn.ParameterList()

        kernel = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        kernel.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

        self.kernel_parameters = Parameter(kernel.weight)
        # todo add bias parameters ? tensor of size out_nc
        self.kernel_bias = None

    def forward(self, ftn_layer):
        # print(self.kernel_parameters)
        kernel_weights = ftn_layer(self.kernel_parameters)
        return kernel_weights


############ BLOCK ###########


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
        x = self.relu(self.bn(functional.conv2d(x, conv_layer_parameters, padding=self.padding)))

        # x = self.relu(self.bn(self.conv(x)))

        return x + residual


############ Primary Net ############

class FTN_Resnet(nn.Module):
    def __init__(self, alpha, input_channels=3, num_layers=1):
        super(FTN_Resnet, self).__init__()
        # TODO delete it
        self.old_params = {}

        self.num_layers = num_layers

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=64,
                                             kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        self.ftn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.ftn_layers.append(FTN.FTNBlock(alpha=alpha, in_nc=64, out_nc=64))

        # todo change to bias True? change the numbers of layers when it fixed
        self.blocks = self._make_blocks(conv_ftn_block, alpha, kernel_size=3, num_channels=64, num_of_layers=num_layers,
                                        bias=False)

        self.kernels = nn.ModuleList()
        for i in range(num_layers):
            self.kernels.append(Kernels())

        self._init_weights()

        self.output = nn.Conv2d(in_channels=64, out_channels=input_channels, kernel_size=3, stride=1, padding=1)

    def _init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if 0 <= m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif -clip_b < m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_blocks(self, block, alpha, kernel_size, num_channels, num_of_layers, padding=1, bias=False):

        layers = [block(in_channels=num_channels, out_channels=num_channels, alpha=alpha, kernel_size=kernel_size,
                        padding=padding,
                        bias=bias, ftn_layer=self.ftn_layers[i]) for i in range(num_of_layers)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(self.num_layers):
            kernel_weights = self.ftn_layers[i](self.kernels[i].kernel_parameters)
            x = self.blocks[i](x, kernel_weights)

        return self.output(x)

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        # todo use cpu only when running on my PC!!
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])

    def get_ftn(self):
        return self.ftn_layers

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
