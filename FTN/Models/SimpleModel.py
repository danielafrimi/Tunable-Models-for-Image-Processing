import torch
import torch.nn.functional as F
from torch import nn
from Models import FTN


def get_conv_layer_with_updated_weights(conv_layer_parameters, input_channels, output_channels):

    # We dont want to learn this layer, we only use it on the input feature map
    generated_conv = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1, padding_mode='zeros')\
        .requires_grad_(False)
    # Load parameters into the layer
    generated_conv.weight = nn.Parameter(conv_layer_parameters)
    # todo what do we need to do with the bias?
    # generated_conv1.bias = nn.Parameter(initial_param['bias'])
    return generated_conv


class SimpleModel(nn.Module):

    def __init__(self, input_number_channels=3, output_number_channels=60):
        """

        :param input_number_channels: input number of channels
        :param output_number_channels: output number of channels
        :param filters_number:
        :param blocks_number: number of the residual blocks
        :param norm_type:
        :param activaion_function_type: activation function type
        :param res_scale:
        :param upsample_mode:
        """
        super(SimpleModel, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() is not None else 'cpu')

        self.input_number_channels = input_number_channels
        self.output_number_channels = output_number_channels

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, padding_mode='zeros')

        self.ftn1 = FTN.FTNBlock(alpha=0, in_nc=64, out_nc=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, padding_mode='zeros')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, padding_mode='zeros')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(3, 3), padding=1,  padding_mode='zeros')

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        # todo The conv_layer_parameters is tensor of size torch.Size([60, 3, 3, 3])
        conv_layer_parameters = self.ftn1(self.conv1.weight)

        # Creating a convolution layer for operating the previous feature map
        generated_conv = get_conv_layer_with_updated_weights(conv_layer_parameters,
                                                             self.input_number_channels, self.output_number_channels)
        x = F.relu(generated_conv(x))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))

        return x

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])

    def __repr__(self):
        return 'Simple_Net'
