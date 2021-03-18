from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms, datasets
import pytorch_lightning as pl
import os


class CifarDataset(pl.LightningDataModule):

    def prepare_data(self, *args, **kwargs):
        CIFAR10(os.getcwd(), train=True, download=True)
        CIFAR10(os.getcwd(), train=False, download=True)

    def train_dataloader(self):
        trainset = CIFAR10(os.getcwd(), train=True, download=False)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

        return trainloader

    def test_dataloader(self):
        testset = CIFAR10(os.getcwd(), train=False, download=False)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

        return testloader

