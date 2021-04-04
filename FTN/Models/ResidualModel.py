import torch
from torch import nn
import FTN.Models.Block as Block


class DenoisingModel(nn.Module):

    def __init__(self, input_number_channels=3, output_number_channels=3, filters_number=64, blocks_number=5,
                 norm_type='adafm', activaion_function_type='relu', res_scale=1):
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
        super(DenoisingModel, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() is not None else 'cpu')

        norm_layer = Block.get_norm_layer(norm_type)

        # We first use this conv layer (before residual blocks)
        fea_conv = Block.conv_block(in_nc=input_number_channels, out_nc=filters_number, stride=2, kernel_size=3,
                                    norm_layer=None,
                                    act_type=None)

        # Creating residual blocks
        # todo this is the only place we use adafm block, maybe insert ftn layer instead?
        resnet_blocks = [
            Block.ResNetBlock(input_number_channels=filters_number, middle_number_channels=filters_number,
                              output_number_channels=filters_number, norm_layer=norm_layer, res_scale=res_scale)
            for _ in range(blocks_number)]

        # todo check if i need this
        LR_conv = Block.conv_block(filters_number, filters_number, kernel_size=3, norm_layer=norm_layer, act_type=None)

        # Up sampling
        upsample_block = Block.upconv_blcok

        # todo check if this is proper for denoising task also
        upsampler = upsample_block(filters_number, filters_number)

        # At the end of the model we have got two conv layers
        HR_conv0 = Block.conv_block(filters_number, filters_number, kernel_size=3, norm_layer=None,
                                    act_type=activaion_function_type)
        HR_conv1 = Block.conv_block(filters_number, output_number_channels, kernel_size=3, norm_layer=None,
                                    act_type=None)

        # self.shorcut_block = Block.ShortcutBlock(Block.sequential(*resnet_blocks, LR_conv))

        self.model = Block.sequential(fea_conv, Block.ShortcutBlock(Block.sequential(*resnet_blocks, LR_conv)),
                                      upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
