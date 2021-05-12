import argparse
import os
import shutil

from torch.utils.data import DataLoader

from Data.HRDataset import HRDataset
from Models.FTN_Resnet import FTN_Resnet
from Models.ResidualModel import DenoisingModel
from Train import Trainer
import wandb
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()

    # tensorboard
    p.add_argument('--log_dir', type=str, default='runs', help='directory for tensorboard logs (common to many runs)')

    # opt
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--noise_std', type=float, default=0.4)
    p.add_argument('--data_path', type=str, default='/cs/labs/werman/daniel023/Lab_vision/FTN/dataset/DIV2K_train_HR')
    args = p.parse_args()
    return args


def make(config):
    path_dataset = '/Users/danielafrimi/Desktop/University/Lab_Vision/FTN/dataset/DIV2K_train_HR'
    # path_dataset = '/cs/labs/werman/daniel023/Lab_vision/FTN/dataset/DIV2K_train_HR'

    # Make the data
    trainset = HRDataset(config.noise_std, dataroot=path_dataset)
    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)

    model = FTN_Resnet(alpha=config.alpha, num_layers=config.layers)
    # model = DenoisingModel()

    # Define the Optimizer and the loss function for the model
    ftn_layers = model.get_ftn()
    optimizers = [torch.optim.Adam(ftn_layer.parameters(), lr=config.learning_rate, betas=(0.5, 0.999)) for ftn_layer in
                  ftn_layers]
    # For the primary model
    optimizers.append(torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999)))

    criterion = nn.L1Loss()

    return model, train_loader, criterion, optimizers


def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, criterion, optimizers = make(config)

        # FIRST STEP
        denoising_trainer = Trainer(train_loader=train_loader, model=model, finetune=False, load=False, CUDA=False,
                                    criterion=criterion, optimizers=optimizers, config=config)
        denoising_trainer.train()



def main():
    args = parse_args()
    wandb.login()

    config = dict(
        epochs=30,
        layers=10,
        batch_size=16,
        learning_rate=0.001,
        dataset="DIV2K",
        noise_std=0.2,
        alpha=0,
        path_dataset='/Users/danielafrimi/Desktop/University/Lab_Vision/FTN/dataset/DIV2K_train_HR',
        architecture="FTN")

    model_pipeline(config)


# TODO
#  6. lr scheulder


if __name__ == '__main__':
    main()
