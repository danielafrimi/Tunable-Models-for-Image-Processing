import torch
from torch import nn
import FTN.Models.Block as Block


class DenoisingModel(nn.Module):

    def __init__(self, input_number_channels=3, output_number_channels=3, filters_number=64, blocks_number=5,
                 norm_type='basic', activaion_function_type='relu', res_scale=1, upsample_mode='upconv'):
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
        fea_conv = Block.conv_block(input_number_channels, out_nc=filters_number, stride=2, kernel_size=3,
                                    norm_layer=None,
                                    act_type=None)

        # Creating residual blocks
        resnet_blocks = [
            Block.ResNetBlock(input_number_channels=filters_number, middle_number_channels=filters_number,
                              output_number_channels=filters_number, norm_layer=norm_layer,
                              act_type=activaion_function_type,
                              res_scale=res_scale)
            for _ in range(blocks_number)]

        LR_conv = Block.conv_block(filters_number, filters_number, kernel_size=3, norm_layer=norm_layer, act_type=None)

        # Up sampling - according to the model
        if upsample_mode == 'upconv':
            upsample_block = Block.upconv_blcok
        # elif upsample_mode == 'pixelshuffle':
        #     upsample_block = Block.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        # todo check if this is proper for RS task only
        upsampler = upsample_block(filters_number, filters_number, act_type=activaion_function_type)

        # At the end of the model we have got two conv layres
        HR_conv0 = Block.conv_block(filters_number, filters_number, kernel_size=3, norm_layer=None,
                                    act_type=activaion_function_type)
        HR_conv1 = Block.conv_block(filters_number, output_number_channels, kernel_size=3, norm_layer=None,
                                    act_type=None)

        self.model = Block.sequential(fea_conv, Block.ShortcutBlock(Block.sequential(*resnet_blocks, LR_conv)),
                                      upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
