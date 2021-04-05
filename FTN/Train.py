import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image

from FTN.Models.ResidualModel import DenoisingModel
from FTN.Models.SimpleModel import SimpleModel
from torch.optim.lr_scheduler import StepLR

class Trainer:

    def __init__(self, train_loader, noise_std, log_dir, lr, batch_size=16, ):
        self.net = DenoisingModel()
        # self.net = SimpleModel()
        self.train_loader = train_loader

        # std of a Gaussian for noising the images
        self.noise_std = noise_std

        # run tensorboard --logdir=runs on terminal
        self.writer = SummaryWriter(log_dir=log_dir)

        # TODO change to lr_schedulers
        self.lr = lr
        self.batch_size = batch_size

        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            print("Using " + str(num_devices) + " GPU's")
            for i in range(num_devices):
                print(torch.cuda.get_device_name(i))
            self.residual_net = nn.DataParallel(self.residual_net)

    def train(self, num_epochs=50):
        """
        Train the model on the data in the data loader and save the weights of the model
        :return: None
        """
        print("Start Training with lr={}, batch_size={}".format(self.lr, self.batch_size))

        # Define the Optimizer and the loss function for the model
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999))
        criterion = nn.L1Loss()
        # scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

        net_loss_per_batch = list()

        # Train the model
        for epoch in range(num_epochs):
            print("Epoch number: {}".format(epoch))
            running_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                # todo normalize the data?
                images, noisy_images = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(noisy_images)
                print(outputs.shape)

                # Calculating loss
                loss = criterion(outputs, images)
                loss.backward()

                optimizer.step()
                # scheduler.step()

                # Visualize
                self.writer.add_image('denoising_images', make_grid(outputs / 2 + 0.5), i)
                self.writer.add_image('real_images', make_grid(images / 2 + 0.5), i)
                self.writer.add_image('noisy_images', make_grid(noisy_images / 2 + 0.5), i)

                # print statistics
                running_loss += loss.item()
                # print("this is the lr {}".format(scheduler.get_last_lr()))

                if i % self.batch_size == self.batch_size - 1:
                    # Save images
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
