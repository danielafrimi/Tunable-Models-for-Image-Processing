import torch
from torch.utils.data import DataLoader

from Data.HRDataset import HRDataset
from Models.FTN_Resnet import FTN_Resnet
from torchvision.utils import make_grid, save_image
from Utils import imshow
# todo take a specific image

path_dataset = '/Users/danielafrimi/Desktop/University/Lab_Vision/FTN/dataset/DIV2K_train_HR'


trainset = HRDataset(noise_level=0.6, dataroot=path_dataset, crop=False)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True)

clean_image, noisy_image = next(iter(trainloader))

# noise = clean_image + torch.rand_like(clean_image)

# imshow(noise, is_grid=False)

save_image(clean_image, fp="clean_image.jpeg")
save_image(noisy_image, fp="noisy_image.jpeg")

alpha_factors = [0, 0.2, 0.5, 0.8, 1]
denoising_images = list()

print("Done saving")
for alpha in alpha_factors:
    model = FTN_Resnet(alpha=alpha, num_layers=5)
    model.load('denoising_model_model_FTN_RESNET_std_0.2_lr_0.001_batch_size_16_good.ckpt')
    # Load model weights
    denoised_image = model(noisy_image)
    save_image(denoised_image, fp="check_{}.jpeg".format(alpha))
    print("done with {}".format(alpha))
    denoising_images.append(denoised_image)

    # todo save and plot