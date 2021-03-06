import argparse
import os
import shutil

from torch.utils.data import DataLoader

from Data.HRDataset import HRDataset
from Models.FTN_Resnet import FTN_Resnet
from Train import Trainer


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


def main():
    args = parse_args()
    run_name = 'lr_denoising_{}_noise_std_{}'.format(args.lr, args.noise_std)
    print("This is the run name {}".format(run_name))

    # Create a directory with log name
    args.log_dir = os.path.join(args.log_dir, run_name)
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)

    # path_dataset = '/Users/danielafrimi/Desktop/University/Lab_Vision/FTN/dataset/DIV2K_train_HR'
    # path_dataset = args.data_path
    path_dataset = '/cs/labs/werman/daniel023/Lab_vision/FTN/dataset/DIV2K_train_HR'

    model = FTN_Resnet(alpha=1, num_layers=5)
    print("FTN RESNET Created with {} num layers on {} noise level ".format(model.num_layers, 0.5))

    del args.data_path

    # Finetune
    trainset = HRDataset(noise_level=0.5, dataroot=path_dataset)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

    denoising_trainer = Trainer(trainloader, model=model, **args.__dict__, finetune=True, load=True, CUDA=True,
                                num_layer=5)
    denoising_trainer.train()


if __name__ == '__main__':
    main()
