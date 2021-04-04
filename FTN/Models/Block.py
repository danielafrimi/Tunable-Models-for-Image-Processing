import torch.nn as nn
from collections import OrderedDict
import functools


class ShortcutBlock(nn.Module):
    # Element-wise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

def get_norm_layer(norm_type):
    # helper selecting normalization layer
    if norm_type == 'batch':
        layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'basic':
        layer = functools.partial(Basic)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, pad_type='zero', norm_layer=None,
               act_type='relu'):
    '''
    Conv layer with padding, normalization, activation
    '''
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    conv_layer = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                           bias=bias,
                           groups=groups)

    # TODO For be more generic we can use a function to peak to activation map
    activation_function = nn.ReLU(inplace=True) if act_type else None

    norm_layer = norm_layer(out_nc) if norm_layer else None

    # Like in the paper, first conv, AdaFM, Relu
    return sequential(p, conv_layer, norm_layer, activation_function)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                 pad_type='zero', norm_layer=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                      pad_type=pad_type, norm_layer=norm_layer, act_type=act_type)
    return sequential(upsample, conv)


class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    '''

    def __init__(self, input_number_channels, middle_number_channels, output_number_channels, kernel_size=3, stride=1,
                 dilation=1, groups=1, bias=True, pad_type='zero', norm_layer=None, act_type='relu', res_scale=1):
        super(ResNetBlock, self).__init__()

        # The resnet block consist of 2 conv layers with relu (with skip connection at the end)
        conv0 = conv_block(input_number_channels, middle_number_channels, kernel_size, stride, dilation, groups, bias,
                           pad_type, norm_layer, act_type)

        # TODO Instead of norm type to be basic, we want do change it to ftn layer?
        conv1 = conv_block(middle_number_channels, output_number_channels, kernel_size, stride, dilation, groups, bias,
                           pad_type, norm_layer, act_type=None)

        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class Basic(nn.Module):

    def __init__(self, in_channel):
        super(Basic, self).__init__()
        self.in_channel = in_channel

    def forward(self, x):
        return x
