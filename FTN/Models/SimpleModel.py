import torch
import torch.nn.functional as F
from torch import nn
from FTN.Models import FTN


class SimpleModel(nn.Module):

    def __init__(self, input_number_channels=3, output_number_channels=3):
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

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=60, kernel_size=3, padding=1, padding_mode='zeros')

        self.ftn1 = FTN.FTNBlock(alpha=0, in_nc=3, out_nc=60)

        self.conv2 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=(3, 3), padding=1, groups=3, padding_mode='zeros')
        self.conv3 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=(3, 3), padding=1, groups=3, padding_mode='zeros')
        self.conv4 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=(3, 3), padding=1, groups=3, padding_mode='zeros')
        self.conv5 = nn.Conv2d(in_channels=60, out_channels=3, kernel_size=(3, 3), padding=1, groups=1, padding_mode='zeros')

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        # todo check - maybe
        conv_layer_parameters = self.ftn1(self.conv1.weight)

        # We dont want to learn this layer, only use it on the input feature map
        generated_conv = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=(3, 3)).requires_grad_(False)

        # Load parameters into the layer
        generated_conv.weight = nn.Parameter(conv_layer_parameters)
        # generated_conv1.bias = nn.Parameter(initial_param['bias'])
        x = generated_conv(x)

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
