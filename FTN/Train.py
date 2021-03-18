import torch
import torch.nn as nn
from torchvision.utils import make_grid

from Model import DenoisingModel
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, train_loader, noise_std, log_dir, batch_size=64, lr=0.0001):
        self.net = DenoisingModel()
        self.train_loader = train_loader
        self.noise_std = noise_std
        self.writer = SummaryWriter(log_dir=log_dir)
        self.lr = lr
        self.batch_size = batch_size


    def train(self, num_epochs=50, plot_net_error=False):
        """
        Train the model on the data in the data loader and save the weights of the model
        :return: None
        """
        print("Start Training with lr={}, batch_size= {}".format(self.lr, self.batch_size))

        # Define the Optimizer and the loss function for the model
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss()

        net_loss_per_batch = list()

        # Train the model
        for epoch in range(num_epochs):
            running_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):

                images, labels = data

                noisy_image = torch.normal(0, self.noise_std, size=images.shape) + images

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(noisy_image)

                # Calculating loss
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()

                # Visualize Reconstructions
                self.writer.add_image('denoising_images', make_grid(outputs), epoch)
                self.writer.add_image('real_images', make_grid(images), epoch)

                # print statistics
                running_loss += loss.item()
                if i % self.batch_size == self.batch_size - 1:  # print every batch
                    net_loss_per_batch.append((running_loss / self.batch_size))
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / self.batch_size))
                    running_loss = 0.0

        if plot_net_error:
            print(" THIS ARE THE ERRORS {} ".format(net_loss_per_batch))
            # visualizer.plot_net_error(net_loss_per_batch, lr, save_img=True)

        print('Finished Training')

        # Save the weights of the trained model
        self.net.save('./denoising_model_std_0.2.ckpt')
