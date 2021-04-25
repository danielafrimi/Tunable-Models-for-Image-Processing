import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image

from Models.SimpleModel import SimpleModel
from Models.Resnet import Resnet
from Models import FTN
from Models.FTN_Resnet import FTN_Resnet
from torch.optim.lr_scheduler import StepLR
from piq import psnr
from Utils import plot_train_data, plot_grad_flow
from torchviz import make_dot


class Trainer:

    def __init__(self, train_loader, noise_std, net, log_dir, lr, batch_size=16, load=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() is not None else 'cpu')
        print("Using {}".format(self.device))

        self.net = net
        if load:
            print("Loading network weights")
            self.net.load('./denoising_model_model_{}_std_{}_lr_{}_batch_size_{}.ckpt'
                          .format(self.net.__repr__(), self.noise_std, self.lr, self.batch_size))
            print("Loaded Succeed")

        self.train_loader = train_loader

        # std of a Gaussian for noising the images
        self.noise_std = noise_std

        self.num_layers = 10

        # run tensorboard --logdir=runs on terminal
        self.writer = SummaryWriter(log_dir=log_dir)

        self.lr = lr
        self.batch_size = batch_size

        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            print("Using " + str(num_devices) + " GPU's")
            for i in range(num_devices):
                print(torch.cuda.get_device_name(i))
            self.net = nn.DataParallel(self.net)

    def train(self, num_epochs=70):
        """
        Train the model on the data in the data loader and save the weights of the model
        :return: None
        """
        print("Start Training with lr={}, batch_size={}".format(self.lr, self.batch_size))

        # Define the Optimizer and the loss function for the model
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999))
        criterion = nn.L1Loss()
        ftn_optimizers = self.get_optimizers_ftn_laer()

        net_loss_per_batch = list()
        net_psnr_per_batch = list()
        # todo when using cuda
        # self.net.to(self.device)

        # Train the model
        for epoch in range(num_epochs):
            print("Epoch number: {}".format(epoch))
            running_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                images, noisy_images = data
                #  todo when using cuda
                # images, noisy_images = images.to(self.device), noisy_images.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                self.net.save_params()

                # forward + backward + optimize
                outputs = self.net(noisy_images)

                # Calculating loss
                loss = criterion(outputs, images)

                loss.backward()

                optimizer.step()
                for optimizer_ftn in ftn_optimizers:
                    optimizer_ftn.step()

                # plot_grad_flow(self.net.named_parameters())
                print("Daniel")
                self.net.check_params()

                # Visualize
                self.writer.add_image('denoising_images', make_grid(outputs), i)
                self.writer.add_image('real_images', make_grid(images), i)
                self.writer.add_image('noisy_images', make_grid(noisy_images), i)

                running_loss += loss.item()
                if i % self.batch_size == self.batch_size - 1:
                    # Saving checkpoint of the network
                    print("Saving network weights - denoising_model_model_{}_std_{}_lr_{}_batch_size_{}" \
                          .format(self.net.__repr__(), self.noise_std, self.lr, self.batch_size))

                    self.net.save('./denoising_model_model_{}_std_{}_lr_{}_batch_size_{}.ckpt'
                                  .format(self.net.__repr__(), self.noise_std, self.lr, self.batch_size))

                    # The larger the value of PSNR, the more efficient is a corresponding compression or filter method
                    self.writer.add_scalar('psnr', psnr(images, torch.clamp(outputs, min=0., max=1.)), epoch)

                    # Save images
                    save_image(make_grid(images), fp="images.jpeg")
                    save_image(make_grid(noisy_images), fp="noisy_images.jpeg")
                    save_image(make_grid(outputs), fp="denoising_images.jpeg")

                    # Visualize on tensorboard
                    self.writer.add_image('denoising_images', make_grid(outputs), epoch)
                    self.writer.add_image('real_images', make_grid(images), epoch)
                    self.writer.add_image('noisy_images', make_grid(noisy_images), epoch)

                    net_loss_per_batch.append((running_loss / self.batch_size))
                    net_psnr_per_batch.append(psnr(images, torch.clamp(outputs, min=0., max=1.)).data.cpu())

                    plot_train_data(net_psnr_per_batch, ["PSNR", 'Number of Mini-Batches'], name='psnr')

                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / self.batch_size))
                    self.writer.add_scalar("loss", (running_loss / self.batch_size), epoch)
                    running_loss = 0.0

        print('Finished Training')

        plot_train_data(net_psnr_per_batch, ["PSNR", 'Number of Mini-Batches'], name='psnr')

        # Save the weights of the trained model
        print("Saving network weights")
        self.net.save('./denoising_model_model_{}_std_{}_lr_{}_batch_size_{}.ckpt'
                      .format(self.net.__repr__(), self.noise_std, self.lr, self.batch_size))

    def get_optimizers_ftn_laer(self):
        ftn_layers = self.net.get_ftn()

        optimizers = [torch.optim.Adam(ftn_layer.parameters(), lr=self.lr, betas=(0.9, 0.999)) for ftn_layer in ftn_layers]
        return optimizers
