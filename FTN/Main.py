import argparse
import os
import shutil

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Data.HRDataset import HRDataset
from Train import Trainer


def parse_args():
    p = argparse.ArgumentParser()

    # tensorboard
    p.add_argument('--log_dir', type=str, default='runs', help='directory for tensorboard logs (common to many runs)')

    # opt
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--noise_std', type=float, default=0.3)

    args = p.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run_name = 'lr_denoising_{}_std_noise_{}'.format(args.lr, args.noise_std)

    # Create a directory with log name
    args.log_dir = os.path.join(args.log_dir, run_name)
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    path_dataset = '/Users/danielafrimi/Desktop/University/Lab_Vision/FTN/dataset/DIV2K_train_HR'

    trainset = HRDataset(args.noise_std, dataroot=path_dataset)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

    denoising_trainer = Trainer(trainloader, **args.__dict__, load=False)
    denoising_trainer.train()
