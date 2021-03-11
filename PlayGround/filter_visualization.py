import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import argparse
from torchvision import models, transforms



# load the model
model = models.resnet50(pretrained=True)
print(model)

# we will save the conv layer weights in this list
model_weights = []

# we will save the 49 conv layers in this list
# it is important to remember that the ResNet-50 model has 50 layers in total.
# 49 of those layers are convolutional layers and a final fully connected layer.
conv_layers = []

# get all the model children as list
model_children = list(model.children())

# counter to keep count of the conv layers
counter = 0
# Append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    # whether any of the Bottleneck layer inside the Sequential blocks contain any convolutional layers.
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolutional layers: {counter}")

# take a look at the conv layers and the respective weights
for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

# visualize the first conv layer filters
plt.figure(figsize=(20, 17))
# Iterating through the weights of the first convolutional laye
for i, filter in enumerate(model_weights[0]):
    plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    plt.imshow(filter[0, :, :].detach(), cmap='gray')
    plt.axis('off')
    plt.savefig('/outputs/filter.png')
plt.show()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='path to image')
    args = vars(ap.parse_args())
