import argparse
import os
import time

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid

import utils
import wandb
# from network import ImageTransformNet
from network_ftn import ImageTransformNet
from vgg import Vgg16

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Global Variables
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 2
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7


def train(args, hyperparameters):
    with wandb.init(project="pytorch-demo", config=hyperparameters):

        config = wandb.config

        device = torch.device('cuda' if torch.cuda.is_available() is not None else 'cpu')

        # visualization of training
        if config.visualize:
            img_transform_512 = transforms.Compose([
                transforms.Resize(512),  # scale shortest side to image_size
                transforms.CenterCrop(512),  # crop center image_size out
                transforms.ToTensor(),  # turn image from [0-255] to [0-1]
                utils.normalize_tensor_transform()  # normalize with ImageNet values
            ])

            # testImage_amber = utils.load_image("content_imgs/amber.jpg")
            # testImage_amber = img_transform_512(testImage_amber)
            # testImage_amber = testImage_amber.repeat(1, 1, 1, 1).detach().cpu() if not config.CUDA \
            #     else testImage_amber.repeat(1, 1, 1, 1).to(device)

            testImage_dan = utils.load_image("content_imgs/dan.jpg")
            testImage_dan = img_transform_512(testImage_dan)
            testImage_dan = testImage_dan.repeat(1, 1, 1, 1).detach().cpu()

            # testImage_maine = utils.load_image("content_imgs/maine.jpg")
            # testImage_maine = img_transform_512(testImage_maine)
            # testImage_maine = testImage_maine.repeat(1, 1, 1, 1) if not config.CUDA \
            #     else testImage_maine.repeat(1, 1, 1, 1).to(device)

        # define network
        image_transformer = ImageTransformNet()
        image_transformer if not config.CUDA else image_transformer.to(device)

        optimizer = Adam(image_transformer.parameters(), LEARNING_RATE)

        loss_mse = torch.nn.MSELoss()

        # load vgg network
        vgg = Vgg16()
        vgg if not config.CUDA else vgg.to(device)

        # get training dataset
        dataset_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),  # scale shortest side to image_size
            transforms.CenterCrop(IMAGE_SIZE),  # crop center image_size out
            transforms.ToTensor(),  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()  # normalize with ImageNet values
        ])

        train_dataset = datasets.ImageFolder(args.dataset, dataset_transform)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

        # style image
        style_transform = transforms.Compose([
            transforms.ToTensor(),  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()  # normalize with ImageNet values
        ])

        style = utils.load_image(args.style_image)
        style = style_transform(style)
        style = style.repeat(config.batch_size, 1, 1, 1) if not config.CUDA else style.repeat(config.batch_size, 1, 1,
                                                                                              1).to(device)
        style_name = os.path.split(args.style_image)[-1].split('.')[0]

        # calculate gram matrices for style feature layer maps we care about
        style_features = vgg(style)
        style_gram = [utils.gram(fmap) for fmap in style_features]

        wandb.watch(image_transformer, loss_mse, log="all", log_freq=5)

        iter_number = 0
        for e in range(EPOCHS):

            # track values for...
            img_count = 0
            aggregate_style_loss = 0.0
            aggregate_content_loss = 0.0
            aggregate_tv_loss = 0.0

            # train network
            image_transformer.train()
            for batch_num, (x, label) in enumerate(train_loader):
                iter_number += 1

                img_batch_read = len(x)
                img_count += img_batch_read

                # zero out gradients
                optimizer.zero_grad()

                # input batch to transformer network
                x = x if not config.CUDA else x.to(device)

                y_hat = image_transformer(x)

                # get vgg features
                y_c_features = vgg(x)
                y_hat_features = vgg(y_hat)

                # calculate style loss
                y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
                style_loss = 0.0
                for j in range(4):
                    style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
                style_loss = STYLE_WEIGHT * style_loss
                aggregate_style_loss += style_loss.item()

                # calculate content loss (h_relu_2_2)
                recon = y_c_features[1]
                recon_hat = y_hat_features[1]

                content_loss = CONTENT_WEIGHT * loss_mse(recon_hat, recon)
                aggregate_content_loss += content_loss.item()

                # calculate total variation regularization (anisotropic version)
                # https://www.wikiwand.com/en/Total_variation_denoising
                diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
                diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
                tv_loss = TV_WEIGHT * (diff_i + diff_j)
                aggregate_tv_loss += tv_loss.item()

                # total loss
                total_loss = style_loss + content_loss + tv_loss

                # backprop
                total_loss.backward()
                optimizer.step()

                # print out status message
                if (batch_num + 1) % 10 == 0:
                    status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  agg_tv: {:.6f}  " \
                             "style: {:.6f}  content: {:.6f}  tv: {:.6f} ".format(
                        time.ctime(), e + 1, img_count, len(train_dataset), batch_num + 1,
                                      aggregate_style_loss / (batch_num + 1.0),
                                      aggregate_content_loss / (batch_num + 1.0),
                                      aggregate_tv_loss / (batch_num + 1.0),
                        style_loss.item(), content_loss.item(), tv_loss.item())

                    print(status)

                    wandb.log({"agg_style": aggregate_style_loss / (batch_num + 1.0),
                               "agg_content": aggregate_content_loss / (batch_num + 1.0),
                               "agg_tv": aggregate_tv_loss / (batch_num + 1.0),
                               "style": style_loss.item(),
                               "content": content_loss.item(),
                               "tv": tv_loss.item()
                               }, step=iter_number)

                if (batch_num + 1) % 30 == 0 and config.visualize:
                    image_transformer.eval()

                    if not os.path.exists("visualization"):
                        os.makedirs("visualization")
                    if not os.path.exists("visualization/%s" % style_name):
                        os.makedirs("visualization/%s" % style_name)

                    # outputTestImage_amber = image_transformer(testImage_amber.to(device)).detach().cpu()
                    # amber_path = "visualization/%s/amber_%d_%05d.jpg" % (style_name, e + 1, batch_num + 1)
                    # utils.save_image(amber_path, outputTestImage_amber.data[0])

                    outputTestImage_dan = image_transformer(testImage_dan.to(device)).detach().cpu()
                    dan_path = "visualization/%s/dan_%d_%05d.jpg" % (style_name, e + 1, batch_num + 1)
                    utils.save_image(dan_path, outputTestImage_dan.data[0])

                    # outputTestImage_maine = image_transformer(testImage_maine.to(device)).detach().cpu()
                    # maine_path = "visualization/%s/maine_%d_%05d.jpg" % (style_name, e + 1, batch_num + 1)
                    # utils.save_image(maine_path, outputTestImage_maine.data[0])

                    # wandb.log({"amber": [wandb.Image(make_grid(outputTestImage_amber),
                    #                                  caption="Style-amber")],
                    #            "dan": [wandb.Image(make_grid(outputTestImage_dan),
                    #                                caption="Style-dan")],
                    #            "maine": [wandb.Image(make_grid(outputTestImage_maine),
                    #                                  caption="Style-maine")],
                    #            })

                    wandb.log({"dan": [wandb.Image(make_grid(outputTestImage_dan),
                                                   caption="Style-dan")],
                               })

                    print("images saved")
                    image_transformer.train()

                if (batch_num + 1) % 30 == 0:
                    # Save the Model
                    if not os.path.exists("models"):
                        os.makedirs("models")
                    filename = "models/" + str(style_name) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
                    torch.save(image_transformer.state_dict(), filename)

        # save model
        image_transformer.eval()

        # Save the Model
        if not os.path.exists("models"):
            os.makedirs("models")
        filename = "models/" + str(style_name) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
        torch.save(image_transformer.state_dict(), filename)


def style_transfer(args):
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" % torch.cuda.current_device())

    # content image
    img_transform_512 = transforms.Compose([
        transforms.Resize(512),  # scale shortest side to image_size
        transforms.CenterCrop(512),  # crop center image_size out
        transforms.ToTensor(),  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()  # normalize with ImageNet values
    ])

    content = utils.load_image(args.source)
    content = img_transform_512(content)
    content = content.unsqueeze(0)
    content = Variable(content).type(dtype)

    # load style model
    style_model = ImageTransformNet().type(dtype)
    style_model.load_state_dict(torch.load(args.model_path))

    # process input image
    stylized = style_model(content).cpu()
    utils.save_image(args.output, stylized.data[0])


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train a model to do style transfer")
    train_parser.add_argument("--style-image", type=str, default='/cs/labs/werman/daniel023/Lab_vision/StyleTransfer'
                                                                 '/style_imgs/rain_princess.jpeg',
                              help="path to a style image to train with")

    # train_parser.add_argument("--dataset", type=str, default='/Users/danielafrimi/Desktop/University/Lab_Vision'
    #                                                          '/StyleTransfer/coco', help="path to a dataset")
    #
    train_parser.add_argument("--dataset", type=str, default='/cs/labs/werman/daniel023/Lab_vision/StyleTransfer/coco'
                              , help="path to a dataset")

    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")

    style_parser = subparsers.add_parser("transfer", help="do style transfer with a trained model")
    style_parser.add_argument("--model-path", type=str, required=True,
                              help="path to a pretrained model for a style image")
    style_parser.add_argument("--source", type=str, required=True, help="path to source image")
    style_parser.add_argument("--output", type=str, required=True, help="file name for stylized output image")

    args = parser.parse_args()

    # command
    if args.subcommand == "train":
        print("Training!")
        train(args, config)
    elif args.subcommand == "transfer":
        print("Style transfering!")
        style_transfer(args)
    else:
        print("invalid command")


config = dict(
    epochs=2,
    batch_size=4,
    learning_rate=1e-3,
    dataset="COCO",
    style_weight=1e5,
    content_weight=1e0,
    tv_weight=1e-7,
    image_size=256,
    visualize=True,
    load=False,
    CUDA=True,
    path_dataset='',
    architecture="ImageTransfer")

if __name__ == '__main__':
    main()
