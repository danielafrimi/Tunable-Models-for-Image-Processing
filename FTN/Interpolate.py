from Models.FTN_Resnet import FTN_Resnet

# todo take a specific image
# testset = HRDataset(args.noise_std, dataroot=path_dataset)

image = None


# FIRST TRAIN

alpha_factors = [0, 0.2, 0.5, 0.8, 1]
denoising_images = list()

for alpha in alpha_factors:
    model = FTN_Resnet(alpha=alpha, num_layers=5)
    denoised_image = model(image)
    denoising_images.append(denoised_image)
    # todo save and plot