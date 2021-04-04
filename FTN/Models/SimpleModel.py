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
        :param activaion_function_type: activation funcion type
        :param res_scale:
        :param upsample_mode:
        """
        super(SimpleModel, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() is not None else 'cpu')

        # todo maybe add group to the conv2d,
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, padding_mode='zeros')

        self.ftn1 = FTN.FTNBlock(alpha=0, in_nc=64, out_nc=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=2, padding_mode='zeros')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=2, padding_mode='zeros')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=2, padding_mode='zeros')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(3, 3), padding=1, groups=1, padding_mode='zeros')

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        # todo check
        conv_layer = self.ftn1(self.conv1.weight)

        x = conv_layer(x)
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
