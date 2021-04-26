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



def parse_args():
    p = argparse.ArgumentParser()

    # tensorboard
    p.add_argument('--log_dir', type=str, default='runs', help='directory for tensorboard logs (common to many runs)')

    # opt
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.001)
    # todo add end_noise and start noise
    p.add_argument('--noise_std', type=float, default=0.6)
    p.add_argument('--data_path', type=str, default='/cs/labs/werman/daniel023/Lab_vision/FTN/dataset/DIV2K_train_HR')
    args = p.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run_name = 'lr_denoising_{}_std_noise_{}'.format(args.lr, args.noise_std)

    # Create a directory with log name
    args.log_dir = os.path.join(args.log_dir, run_name)
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)

    path_dataset = '/Users/danielafrimi/Desktop/University/Lab_Vision/FTN/dataset/DIV2K_train_HR'
    # path_dataset = '/cs/labs/werman/daniel023/Lab_vision/FTN/dataset/DIV2K_train_HR'
    # path_dataset = args.data_path
    trainset = HRDataset(args.noise_std, dataroot=path_dataset)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

    ftn_layers = [FTN.FTNBlock(alpha=0.5, in_nc=64, out_nc=64) for i in range(1)]
    net = FTN_Resnet(alpha=0.5, ftn_layers=ftn_layers)
    # net = Resnet()
    # net = SimpleModel()
    print("{} Created".format(net.__repr__()))

    del args.data_path

    denoising_trainer = Trainer(trainloader, net=net, **args.__dict__, load=False)
    denoising_trainer.train()

   #  ftn_resnet = FTN_Resnet(alpha=1)
   #  # todo load weights from the previous Train
   #  # todo change the data
   #  trainset = HRDataset(args.end_noise, dataroot=path_dataset)
   #  trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
   #
   #  denoising_trainer = Trainer(trainloader, net=ftn_resnet, **args.__dict__, load=False)
   #  denoising_trainer.train()
   #
   # # todo start interpolation in another file
