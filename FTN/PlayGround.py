import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision
import torchvision.transforms as transforms
from Models import Model
from FTN.Data.Utils import imshow

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

denoising_net = Model.DenoisingModel()

dataiter = iter(trainloader)
images, labels = dataiter.next()

# output = denoising_net(images)


noisy_image = torch.normal(0, 0.2, size=images.shape) + images

imshow(torchvision.utils.make_grid(noisy_image))




