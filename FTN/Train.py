import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image

from Models.ResidualModel import DenoisingModel
from Models.SimpleModel import SimpleModel
from Models.Resnet import Resnet
from torch.optim.lr_scheduler import StepLR
from Utils import calc_PSNR

class Trainer:

    def __init__(self, train_loader, noise_std, log_dir, lr, batch_size=16, load=True):
        # self.device = torch.device('cuda' if torch.cuda.is_available() is not None else 'cpu')
        # print(self.device)

        # self.net = DenoisingModel()
        # self.net = SimpleModel()
        self.net = Resnet()
        if load:
            self.net.load('./denoising_model_std_0.2.ckpt')

        self.train_loader = train_loader

        # std of a Gaussian for noising the images
        self.noise_std = noise_std

        # run tensorboard --logdir=runs on terminal
        self.writer = SummaryWriter(log_dir=log_dir)

        self.lr = lr
        self.batch_size = batch_size

        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            print("Using " + str(num_devices) + " GPU's")
            for i in range(num_devices):
                print(torch.cuda.get_device_name(i))
            self.residual_net = nn.DataParallel(self.residual_net)

    def train(self, num_epochs=70):
        """
        Train the model on the data in the data loader and save the weights of the model
        :return: None
        """
        print("Start Training with lr={}, batch_size={}".format(self.lr, self.batch_size))

        # Define the Optimizer and the loss function for the model
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999))
        criterion = nn.L1Loss()

        net_loss_per_batch = list()
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

                # forward + backward + optimize
                outputs = self.net(noisy_images)

                # Calculating loss
                loss = criterion(outputs, images)
                loss.backward()

                optimizer.step()

                # Visualize
                self.writer.add_image('denoising_images', make_grid(outputs), i)
                self.writer.add_image('real_images', make_grid(images), i)
                self.writer.add_image('noisy_images', make_grid(noisy_images), i)

                self.writer.add_scalar('psnr', calc_PSNR(images, outputs), i)

                # print statistics
                running_loss += loss.item()

                if i % self.batch_size == self.batch_size - 1:
                    # Saving checkpoint of the network
                    self.net.save('./denoising_model_std_0.2.ckpt')

                    print(calc_PSNR(images, outputs))
                    # The larger the value of PSNR, the more efficient is a corresponding compression or filter method
                    self.writer.add_scalar('psnr', calc_PSNR(images, outputs), i)

                    # Save images
                    # Visualize
                    self.writer.add_image('denoising_images1', make_grid(outputs), i)
                    self.writer.add_image('real_images1', make_grid(images), i)
                    self.writer.add_image('noisy_images1', make_grid(noisy_images), i)
                    save_image(make_grid(images), fp="images.jpeg")
                    save_image(make_grid(noisy_images), fp="noisy_images.jpeg")
                    save_image(make_grid(outputs), fp="denoising_images.jpeg")

                    net_loss_per_batch.append((running_loss / self.batch_size))

                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / self.batch_size))
                    self.writer.add_scalar("loss", (running_loss / self.batch_size), epoch)
                    running_loss = 0.0

        print('Finished Training')

        # Save the weights of the trained model
        self.net.save('./denoising_model_std_0.2.ckpt')
