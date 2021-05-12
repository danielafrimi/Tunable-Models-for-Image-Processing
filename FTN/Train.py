import logging

import torch
import torch.nn as nn
import wandb
from piq import psnr
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from Utils import freeze_network_weights
from Utils import plot_train_data


class Trainer:

    def __init__(self, train_loader, model, config, criterion, optimizers, finetune=False, load=True,
                 CUDA=False, ):
        self.device = torch.device('cuda' if torch.cuda.is_available() is not None else 'cpu')

        self.model = model
        self.GPU = CUDA
        self.num_layers = config.layers

        # std of a Gaussian for noising the images
        self.noise_std = config.noise_std if not finetune else 0.4
        self.finetune = finetune

        self.train_loader = train_loader
        self.lr = config.learning_rate
        self.batch_size = config.batch_size
        self.criterion = criterion
        self.optimizers = optimizers

        self.num_epochs = config.epochs

        logger_file_name = "ftn_layers_{}_lr_{}_noisr_{}_epochs_{}_finetune_{}.log" \
            .format(self.num_layers, self.lr, self.noise_std, self.num_epochs, self.finetune)
        logging.basicConfig(filename=logger_file_name, level=logging.INFO, datefmt='%H:%M:%S')

        self.logger = logging.getLogger('FTN')
        fh = logging.FileHandler(logger_file_name, "w")
        self.logger.addHandler(fh)

        if load:
            self.logger.info("Loading network weights")
            self.model.load('./FTN_RESNET_std_{}_lr_{}_batch_size_{}_epochs_{}_layer_{}_finetune_{}.ckpt'
                            .format(0.2, 0.001, 16, 30, self.num_layers, False))
            self.logger.info("Loaded Succeed")

        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            print("Using " + str(num_devices) + " GPU's")
            for i in range(num_devices):
                print(torch.cuda.get_device_name(i))
            self.model = nn.DataParallel(self.model)

        if finetune:
            self.logger.debug("Freezing Part of Model Weights")
            # print("Freezing Part of Model Weights")
            freeze_network_weights(self.model)

    def train(self):
        """
        Train the model on the data in the data loader and save the weights of the model
        :return: None
        """
        self.logger.info("Start Training with lr={}, batch_size={}".format(self.lr, self.batch_size))
        print("Start Training with lr={}, batch_size={}".format(self.lr, self.batch_size))

        # # Define the Optimizer and the loss function for the model
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        # criterion = nn.L1Loss()

        wandb.watch(self.model, self.criterion, log="all", log_freq=5)

        # ftn_optimizers = self._get_ftn_optimizers()

        loss_per_batch = list()
        PSNR_per_batch = list()

        self.model if not self.GPU else self.model.to(self.device)

        for ftn_layer in self.model.self.model.get_ftn():
            if self.GPU:
                ftn_layer.to(self.device)

        # Train the model
        iter_number = 0
        for epoch in tqdm(range(self.num_epochs)):
            self.logger.info("Epoch number: {}".format(epoch))
            # print("Epoch number: {}".format(epoch))
            running_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                images, noisy_images = data

                images, noisy_images = (images, noisy_images) if not self.GPU else (
                    images.to(self.device), noisy_images.to(self.device))

                # zero the parameter gradients
                for optimizer in self.optimizers: optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(noisy_images)

                # Calculating loss
                loss = self.criterion(outputs, images)

                loss.backward()

                for optimizer in self.optimizers: optimizer.step()

                running_loss += loss.item()
                iter_number += 1
                if i % self.batch_size == self.batch_size - 1:
                    # Saving checkpoint of the network
                    print("Saving network weights - denoising_model_FTN_RESNET_"
                          "std_{}_lr_{}_batch_size_{}_epochs_{}_layer_{}_finetune_{}" \
                          .format(self.noise_std, self.lr, self.batch_size, self.num_epochs, self.num_layers,
                                  self.finetune))

                    self.model.save('./FTN_RESNET_std_{}_lr_{}_batch_size_{}_epochs_{}_layer_{}_finetune_{}.ckpt'
                                    .format(self.noise_std, self.lr, self.batch_size, self.num_epochs, self.num_layers,
                                            self.finetune))

                    # Save images
                    save_image(make_grid(torch.cat([images, noisy_images, outputs])),
                               fp="images batchsize{}_lr_{}_noise_{}_layers_{}.jpeg"
                               .format(self.batch_size, self.lr, self.noise_std, self.num_layers))

                    print(psnr(images, torch.clamp(outputs, min=0., max=1.)).data.cpu())

                    # The larger the value of PSNR, the more efficient is a corresponding compression or filter method
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / self.batch_size))
                    wandb.log({"PSNR": psnr(images, torch.clamp(outputs, min=0., max=1.)).data.cpu(),
                               "loss": float(running_loss / self.batch_size)}, step=iter_number)

                    running_loss = 0.0

        self.logger.info('Finished Training')
        print('Finished Training')

        # Save the weights of the trained model
        self.model.save('./FTN_RESNET_std_{}_lr_{}_batch_size_{}_epochs_{}_layer_{}_finetune_{}.ckpt'
                        .format(self.noise_std, self.lr, self.batch_size, self.num_epochs, self.num_layers,
                                self.finetune))

    def _get_ftn_optimizers(self):
        ftn_layers = self.model.get_ftn()
        if self.GPU:
            for ftn_layer in ftn_layers:
                ftn_layer.to(self.device)

        optimizers = [torch.optim.Adam(ftn_layer.parameters(), lr=self.lr, betas=(0.5, 0.999)) for ftn_layer in
                      ftn_layers]
        return optimizers
