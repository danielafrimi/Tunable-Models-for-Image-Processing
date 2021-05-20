import numpy as np
import torch
import wandb
from piq import psnr
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from Data.HRDataset import HRDataset
from Models.FTN_Resnet import FTN_Resnet

alpha_factors = [0, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
# alpha_factors = [0, 1]

# todo take an image with noise 0.5 and see how with alpha=0 works and alpha=1 (for checking how ftn)


config = dict(
    epochs=30,
    layers=7,
    batch_size=16,
    learning_rate=0.001,
    dataset="DIV2K",
    noise_std=0.5,
    alpha=0,
    finetune=True,
    load=True,
    CUDA=True,
    path_dataset='/Users/danielafrimi/Desktop/University/Lab_Vision/FTN/dataset/DIV2K_train_HR',
    architecture="FTN")

with wandb.init(project="pytorch-demo", config=config):
    config = wandb.config

    # We want to see how the smoothness changes according to the alpha parameter.
    # need to take an image with noise 0.4 or 0.3 and see how it acts.

    path_dataset = '/Users/danielafrimi/Desktop/University/Lab_Vision/FTN/dataset/DIV2K_train_HR'
    # path_dataset = '/cs/labs/werman/daniel023/Lab_vision/FTN/dataset/DIV2K_train_HR'

    trainset = HRDataset(noise_level=0.3, dataroot=path_dataset, crop=True)
    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)

    clean_image, noisy_image = next(iter(trainloader))
    denoising_images_per_alpha = list()

    for i, alpha in enumerate(alpha_factors):
        # Check if after finetune the model still clean images with 0.2 std noise
        model = FTN_Resnet(alpha=alpha, num_layers=config.layers)
        model.load('./FTN_RESNET_std_{}_lr_{}_batch_size_{}_epochs_{}_layer_{}_finetune_{}.ckpt'
                   .format(config.noise_std, config.learning_rate, config.batch_size, 30, config.layers, True))

        denoised_image = model(noisy_image)

        batch_tensors = torch.cat([clean_image, noisy_image, denoised_image])
        denoising_images_per_alpha.append(batch_tensors)

        wandb.log({"images_{}".format(i): [wandb.Image(make_grid(batch_tensors), caption="alpha_{}".format(alpha))]

                   })

        print(i)
        wandb.log({"PSNR": psnr(clean_image, torch.clamp(denoised_image, min=0., max=1.)).data.cpu()}, step=i + 1)
