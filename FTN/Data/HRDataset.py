import numpy as np
import torch
import torch.utils.data as data
import Data.Utils as Utils
import random


# we can use a much smaller training set for image denoising.
# This is because denoising each pixel of an image can be seen as one regression problem

class HRDataset(data.Dataset):
    """
    # Read LR images only in the test phase
    DIV2K dataset - using the HR images only
    """

    def __init__(self, noise_level, crop=True,
                 dataroot='/cs/labs/werman/daniel023/Lab_vision/FTN/dataset/DIV2K_train_HR'):
        super(HRDataset, self).__init__()

        self.paths_LR = None
        self.noise_level = noise_level
        self.crop = crop

        # create array of dataset images
        self.paths_LR = Utils.get_image_paths(dataroot)

    def __getitem__(self, index):

        # get HR image
        HR_path = self.paths_LR[index]
        img_HR = Utils.read_img(HR_path)

        if not self.crop:
            noisy_image = torch.normal(0, self.noise_level, size=img_HR.shape) + img_HR
            return np.transpose(img_HR, (2, 0, 1)), np.transpose(noisy_image, (2, 0, 1))

        # todo modcrop in the validation / test phase
        # if self.opt['phase'] != 'train':
        #     img_HR = Utils.modcrop(img_HR, 2)
        #     img_LR = Utils.modcrop(img_LR, 2)

        H, W, C = img_HR.shape

        # randomly crop
        rnd_h = random.randint(0, max(0, H - 96))
        rnd_w = random.randint(0, max(0, W - 96))
        img_HR = img_HR[rnd_h:rnd_h + 96, rnd_w:rnd_w + 96, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]

        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        noisy_img = torch.normal(0, self.noise_level, size=img_HR.shape) + img_HR
        # sigma / 255.0 * np.random.normal(size=np.shape(self.data))

        return img_HR, torch.clamp(noisy_img, min=0., max=1.)

    def __len__(self):
        return len(self.paths_LR)
