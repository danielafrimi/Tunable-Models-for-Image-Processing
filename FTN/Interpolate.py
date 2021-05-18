import torch
from torch.utils.data import DataLoader

from Data.HRDataset import HRDataset
from Models.FTN_Resnet import FTN_Resnet
from torchvision.utils import make_grid, save_image
from Utils import imshow
# todo take a specific image

path_dataset = '/Users/danielafrimi/Desktop/University/Lab_Vision/FTN/dataset/DIV2K_train_HR'


trainset = HRDataset(noise_level=0.2, dataroot=path_dataset, crop=False)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True)

clean_image, noisy_image = next(iter(trainloader))

# noise = clean_image + torch.rand_like(clean_image)

# imshow(noise, is_grid=False)

save_image(clean_image, fp="clean_image.jpeg")
save_image(noisy_image, fp="noisy_image.jpeg")

alpha_factors = [0, 0.2, 0.5, 0.8, 1]

model = FTN_Resnet(alpha=0, num_layers=5)
model.load('./FTN_RESNET_std_{}_lr_{}_batch_size_{}_epochs_{}_layer_{}_finetune_{}.ckpt'
                    .format(0.2, 0.001, 16, 30, 5,False))

denoised_image = model(noisy_image)

for alpha in alpha_factors:
    model = FTN_Resnet(alpha=alpha, num_layers=5)
    # model.load('./FTN_RESNET_std_{}_lr_{}_batch_size_{}_epochs_{}_layer_{}_finetune_{}.ckpt'
    #                 .format(noise_std, self.lr, self.batch_size, 30, self.num_layers,
    #                         False))
    # Load model weights
    denoised_image = model(noisy_image)
    save_image(denoised_image, fp="check_{}.jpeg".format(alpha))
    print("done with {}".format(alpha))


