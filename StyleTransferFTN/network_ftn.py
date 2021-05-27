import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as functional
from FTN import FTNBlock


class Kernels(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(Kernels, self).__init__()

        kernel = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

        # todo add bias?
        self.kernel_parameters = Parameter(kernel.weight)
        self.kernel_bias = Parameter(kernel.bias)

    def forward(self, ftn_layer):
        kernel_weights = ftn_layer(self.kernel_parameters)
        return kernel_weights


# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, ftn_layer=None):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.ftn_layer = ftn_layer
        self.reflection_pad = nn.ReflectionPad2d(padding)
        # self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.stride = stride
        self.kernel = Kernels(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        if self.ftn_layer is None:
            out = functional.conv2d(out, self.kernel.kernel_parameters, bias=self.kernel.kernel_bias, stride=self.stride)
        else:
            conv_weights = self.ftn_layer(self.kernel.kernel_parameters)
            out = functional.conv2d(out, conv_weights, stride=self.stride)

        # out = self.conv2d(out)
        # print("this is out: ", out.shape)
        return out


# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


# Residual Block
#   adapted from pytorch tutorial
#   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-
#   intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

        # todo create here the numbers of kernels and hold it as parameters_list

    def forward(self, x, conv1_weights=None, conv2_weights=None):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        # out = self.relu(self.in1(functional.conv2d(x, conv1_weights)))
        # out = self.in2(functional.conv2d(out, conv2_weights))
        out = out + residual
        out = self.relu(out)
        return out


class ImageTransformNet(nn.Module):

    def __init__(self, alpha=0):
        super(ImageTransformNet, self).__init__()

        self.kernels = nn.ModuleList()
        self.ftn_layers = nn.ModuleList()

        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        ftn_layer1 = FTNBlock(alpha=alpha, in_nc=3, out_nc=32, groups=1) # todo change the groups
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1, ftn_layer=ftn_layer1)
        # FTN layer
        # self.kernels.append(Kernels(in_channels=3, out_channels=32, kernel_size=9, stride=1))
        # self.ftn_layers.append(FTNBlock(alpha=alpha, in_nc=3, out_nc=32, groups=1))
        self.in1_e = nn.InstanceNorm2d(32, affine=True)

        ftn_layer2 = FTNBlock(alpha=alpha, in_nc=32, out_nc=64, groups=16)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, ftn_layer=ftn_layer2)
        # FTN layer
        # self.kernels.append(Kernels(in_channels=32, out_channels=64, kernel_size=3, stride=2))
        # self.ftn_layers.append(FTNBlock(alpha=alpha, in_nc=32, out_nc=64, groups=16))
        self.in2_e = nn.InstanceNorm2d(64, affine=True)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(128, affine=True)

        # residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True)

        for ftn in self.ftn_layers:
            ftn.init_weights()

    def forward(self, x):
        # encode

        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        # print(self.kernels[0].kernel_parameters.shape)
        # conv1_weights = self.ftn_layers[0](self.kernels[0].kernel_parameters)
        # y = self.relu(self.in1_e(functional.conv2d(x, conv1_weights)))
        # print("this is y shape", y.shape)

        # print(self.kernels[1].kernel_parameters.shape)
        # conv2_weights = self.ftn_layers[1](self.kernels[1].kernel_parameters)
        # y = self.relu(self.in2_e(functional.conv2d(y, conv2_weights)))

        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        # y = self.tanh(self.in1_d(self.deconv1(y)))
        y = self.deconv1(y)

        return y

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        # todo use cpu only when running on my PC!!
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
