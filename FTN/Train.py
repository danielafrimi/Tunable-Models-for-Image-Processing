import torch
import torch.nn as nn
from piq import psnr
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image

from Utils import freeze_network_weights
from Utils import plot_train_data


class Trainer:

    def __init__(self, train_loader, noise_std, model, log_dir=None, lr=0.001, batch_size=16, finetune=False, load=True,
                 CUDA=False, num_layer=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() is not None else 'cpu')
        print("Using {}".format(self.device))

        self.model = model
        self.GPU = CUDA
        self.num_layers = num_layer

        # std of a Gaussian for noising the images
        self.noise_std = noise_std if not finetune else 0.6
        self.finetune = finetune

        if finetune:
            print("Freezing Part of Model Weights")
            freeze_network_weights(self.model)

        self.train_loader = train_loader
        self.lr = lr
        self.batch_size = batch_size

        self.num_epochs = 30 if not finetune else 50

        # run tensorboard --logdir=runs on terminal
        # self.writer = SummaryWriter(log_dir=log_dir)

        if load:
            print("Loading network weights")
            self.model.load('./FTN_RESNET_std_{}_lr_{}_batch_size_{}_epochs_{}_layer_{}_finetune_{}.ckpt'
                            .format(0.1, 0.001, 16, 30, self.num_layers, False))
            print("Loaded Succeed")

        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            print("Using " + str(num_devices) + " GPU's")
            for i in range(num_devices):
                print(torch.cuda.get_device_name(i))
            self.model = nn.DataParallel(self.model)

    def train(self):
        """
        Train the model on the data in the data loader and save the weights of the model
        :return: None
        """
        print("Start Training with lr={}, batch_size={}".format(self.lr, self.batch_size))

        # Define the Optimizer and the loss function for the model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        criterion = nn.L1Loss()
        ftn_optimizers = self._get_ftn_optimizers()

        loss_per_batch = list()
        PSNR_per_batch = list()

        self.model if not self.GPU else self.model.to(self.device)

        # Train the model
        for epoch in range(self.num_epochs):
            print("Epoch number: {}".format(epoch))
            running_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                images, noisy_images = data

                images, noisy_images = (images, noisy_images) if not self.GPU else (
                    images.to(self.device), noisy_images.to(self.device))

                # zero the parameter gradients
                optimizer.zero_grad()
                # self.model.save_params()

                # forward + backward + optimize
                outputs = self.model(noisy_images)

                # Calculating loss
                loss = criterion(outputs, images)

                loss.backward()

                optimizer.step()
                for optimizer_ftn in ftn_optimizers:
                    optimizer_ftn.step()

                # if epoch % 2 == 0:
                #     optimizer.param_groups[0]['lr'] *= 0.9
                #     print(optimizer.param_groups[0]['lr'])

                # self.model.check_params()

                running_loss += loss.item()
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
                    save_image(make_grid(images), fp="clean images batchsize{}_lr_{}_noise_{}_layers_{}.jpeg"
                               .format(self.batch_size, self.lr, self.noise_std, self.num_layers))
                    save_image(make_grid(noisy_images), fp="noisy images batchsize{}_lr_{}_noise_{}_layers_{}.jpeg"
                               .format(self.batch_size, self.lr, self.noise_std, self.num_layers))
                    save_image(make_grid(outputs), fp="denoising images batchsize{}_lr_{}_noise_{}_layers_{}.jpeg"
                               .format(self.batch_size, self.lr, self.noise_std, self.num_layers))

                    loss_per_batch.append((running_loss / self.batch_size))
                    print(psnr(images, torch.clamp(outputs, min=0., max=1.)).data.cpu())

                    # The larger the value of PSNR, the more efficient is a corresponding compression or filter method
                    PSNR_per_batch.append(psnr(images, torch.clamp(outputs, min=0., max=1.)).data.cpu())

                    plot_train_data(PSNR_per_batch, ["PSNR", 'Number of Mini-Batches'], name='psnr')

                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / self.batch_size))
                    running_loss = 0.0

        print('Finished Training')

        plot_train_data(PSNR_per_batch, ["PSNR", 'Number of Mini-Batches'], name='psnr')
        plot_train_data(loss_per_batch, ["Loss", 'Number of Mini-Batches'], name='loss')

        # Save the weights of the trained model
        print("Saving network weights - THE END")
        self.model.save('./FTN_RESNET_std_{}_lr_{}_batch_size_{}_epochs_{}_layer_{}_finetune_{}.ckpt'
                        .format(self.noise_std, self.lr, self.batch_size, self.num_epochs, self.num_layers,
                                self.finetune))

    def train_epoch(self):
        pass

    def _get_ftn_optimizers(self):
        ftn_layers = self.model.get_ftn()
        if self.GPU:
            for ftn_layer in ftn_layers:
                ftn_layer.to(self.device)

        optimizers = [torch.optim.Adam(ftn_layer.parameters(), lr=self.lr, betas=(0.5, 0.999)) for ftn_layer in
                      ftn_layers]
        return optimizers
