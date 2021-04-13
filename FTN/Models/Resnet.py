import torch.nn as nn
import torch


class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=False):
        super(Conv_BN_ReLU, self).__init__()
        if padding == None:
            if stride == 1:
                padding = (kernel_size - 1) // 2
            else:
                padding = 0

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #shape is x torch.Size([16, 64, 96, 96])
        return self.relu(self.bn(self.conv(x)))


class Resnet(nn.Module):
    def __init__(self, input_channels=3):
        super(Resnet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=64,
                                             kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        # todo change to bias True?
        self.blocks = self._make_layers(Conv_BN_ReLU, 3, 64, num_of_layers=10, bias=False)

        self.output = nn.Conv2d(in_channels=64, out_channels=input_channels,
                                kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block, kernel_size, num_channels, num_of_layers, padding=1, groups=1, bias=False):

        layers = [block(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=padding,
                        groups=groups, bias=bias) for _ in range(num_of_layers)]
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.blocks(x)
        return self.output(x) + residual

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])

    def __repr__(self):
        return 'RESNET'
