import torch
from torch import nn


class DenoisingModel(nn.Module):

    def __init__(self):
        super(DenoisingModel, self).__init__()
        # define the archtacture here
        pass

    def forward(self, x):
        pass

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
