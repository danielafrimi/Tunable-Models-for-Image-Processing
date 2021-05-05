import argparse
import os
import shutil

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Data.HRDataset import HRDataset
from Train import Trainer

from Models.SimpleModel import SimpleModel
from Models.Resnet import Resnet
from Models.FTN_Resnet import FTN_Resnet
from Models import FTN
import torch.nn as nn



def parse_args():
    p = argparse.ArgumentParser()

    # tensorboard
    p.add_argument('--log_dir', type=str, default='runs', help='directory for tensorboard logs (common to many runs)')

    # opt
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--noise_std', type=float, default=0.2)
    p.add_argument('--data_path', type=str, default='/cs/labs/werman/daniel023/Lab_vision/FTN/dataset/DIV2K_train_HR')
    args = p.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run_name = 'lr_denoising_{}_noise_std_{}'.format(args.lr, args.noise_std)
    print("This is the run name {}".format(run_name))

    # Create a directory with log name
    args.log_dir = os.path.join(args.log_dir, run_name)
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)

    path_dataset = '/Users/danielafrimi/Desktop/University/Lab_Vision/FTN/dataset/DIV2K_train_HR'
    # path_dataset = args.data_path
    # path_dataset = '/cs/labs/werman/daniel023/Lab_vision/FTN/dataset/DIV2K_train_HR'

    trainset = HRDataset(args.noise_std, dataroot=path_dataset)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

    # todo change it to more formal
    # todo more than 1 gpu
    # ftn_layers = [nn.DataParallel(FTN.FTNBlock(alpha=0, in_nc=64, out_nc=64)) for i in range(num_layers)]
    model = FTN_Resnet(alpha=0, num_layers=5)

    print("'FTN_RESNET' Created with {} num layers on {}".format(model.num_layers, args.noise_std))

    del args.data_path

    # FIRST TRAIN
    denoising_trainer = Trainer(trainloader, model=model, **args.__dict__, finetune=False, load=False, GPU=False)
    denoising_trainer.train()

# TODO
#  1. check if kernels update and ftn not with alpoha=0. the same with alpha=1  - WORKS!
#  2.check if freezing network works - WORKS!
#  3. run experiments with larger number of epochs in finetune with sbatch
#  4. filter as identity - to initialize them at the beginning that way

