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
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--noise_std', type=float, default=0.5)
    p.add_argument('--data_path', type=str, default='/cs/labs/werman/daniel023/Lab_vision/FTN/dataset/DIV2K_train_HR')
    args = p.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run_name = 'lr_denoising_{}_noise_std_{}'.format(args.lr, args.noise_std)
    # print("This is the run name {}".format(run_name))


    # Create a directory with log name
    # args.log_dir = os.path.join(args.log_dir, run_name)
    # if os.path.exists(args.log_dir):
    #     shutil.rmtree(args.log_dir)

    # path_dataset = '/Users/danielafrimi/Desktop/University/Lab_Vision/FTN/dataset/DIV2K_train_HR'
    # path_dataset = args.data_path
    path_dataset = '/cs/labs/werman/daniel023/Lab_vision/FTN/dataset/DIV2K_train_HR'

    trainset = HRDataset(args.noise_std, dataroot=path_dataset)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

    model = FTN_Resnet(alpha=0, num_layers=10)

    print("FTN_RESNET Created with {} layers on noise {}".format(model.num_layers, args.noise_std))

    del args.data_path

    # FIRST STEP
    denoising_trainer = Trainer(trainloader, model=model, **args.__dict__, finetune=False, load=False, CUDA=True,
                                num_layer=10)
    denoising_trainer.train()

# TODO
#  3. run experiments with larger number of epochs in finetune with sbatch
#  6. lr scheulder



