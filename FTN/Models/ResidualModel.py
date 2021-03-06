import torch
from torch import nn
import Models.Block as Block


class DenoisingModel(nn.Module):

    def __init__(self, input_number_channels=3, output_number_channels=3, filters_number=64, blocks_number=5,
                 norm_type='basic', activaion_function_type='relu', res_scale=1):
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
        self._init_weights()

        self.model = Block.sequential(fea_conv, Block.ShortcutBlock(Block.sequential(*resnet_blocks, LR_conv)),
                                      upsampler, HR_conv0, HR_conv1)

    def _init_weights(self):
        for m in self.modules():
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

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
