# Lab - Tunable Models for Image Processing

## Introduction

Deep convolutions neural network has demonstrated its capability of learning a deterministic mapping for the desired imagery effect. However, the large variety of user flavors motivates the possibility of continuous transition among different output effects. \\

For example if our task is image denoisng, the most common method is by supervised models (deep learning), when the input is a noisy image, and the output will be the clean image (fixed pre-determined corruption level). However, models are optimized for only a single degradation level, so if we have a batch of noisy images with different Gaussian noise, our model will not clean the images well (in reality, the degradation level is not known and at inference time, the model can under-perform). \\ 

In order to overcome this limitation, continuous-level based models have been proposed. in such models, the output image is based on a target parameter which can be adjusted at inference time. 

## Implement Smoother Network Tuning and Interpolation for Continuous-level Image Processing paper
To get a better intuition about Continuous-level models, I implemented https://arxiv.org/abs/1904.08118 - Modulating Image Restoration with Continual Levels via Adaptive Feature Modification Layers (CVPR 2019), and https://arxiv.org/abs/1811.10515 - DNI - Deep Network Interpolation for Continuous Imagery Effect Transition (CVPR 2018)

I have implemented the FTN paper by Samsung Research (2020, with no code reference. The proposed architecture learns kernels during training (taken from the idea in a hyper networks paper).

In the paper they proposed a smoother network tuning and interpolation method for Continuous-level using the FTN, which is structurally smoother than the other frameworks. In addition, the method is comparable in adaptation and interpolation performance on multiple imaging levels, significantly
smoother in practice, and efficient in both memory and computational complexity. 

Filter Transition Network (FTN) which is a non-linear module that easily adapts to new levels (and a method to initialize non- linear CNNs with identity mappings).

### FTN- Densioing Task
 

Train a renset model (with 5-10 layers) on denoising task. added a Gaussian noise with std equal to 0.2 to each image. I've added the FTN blocks after each filter, In the first training alpha=0 (we don't learn the FTN blocks because the gradient is 0). the results were the same to the basic model (PSNR 27-28). 
It can be seen that with the architecture of the model on 16 layers, the model tries to clear a noise of 0.2, but does not maintain the sharpness of the original images.


[comment]: <> (<p float="left">)

[comment]: <> (  <img src="readme_figures/noise0.2_firststep.png" width="200" />)

[comment]: <> (  <img src="readme_figures/psnr_0.2.png" width="200" />)

[comment]: <> (  <img src="readme_figures/loss_0.2noise_first_step.png" width="200"/>)

[comment]: <> (</p>)

<p align="center">

<img src="readme_figures/noise0.2_firststep.png" alt="Denoising" width="70%"/>

</p>

<p align="center">

<img src="readme_figures/psnr_0.2.png" alt="PSNR" width="70%"/>

</p>

<p align="center">

<img src="readme_figures/loss_0.2noise_first_step.png" alt="PSNR" width="70%"/>

</p>



It can be seen that with the architecture of the model on 16 layers, the model tries to clear a noise of 0.4, 
but does not maintain the sharpness of the original images.

<p align="center">

<img src="readme_figures/images_16_layers_noise0.4_.png" alt="16 Layers on 0.4 std noise" width="70%"/>

</p>


[comment]: <> (Clean Images During Training:)

[comment]: <> (<p align="center">)

[comment]: <> (<img src="FTN/results/clean%20images%20batchsize16_lr_0.001_noise_0.2_layers_5.jpeg" alt="Clean Images" width="70%"/>)

[comment]: <> (</p>)

[comment]: <> (Noisy Images During Training:)

[comment]: <> (<p align="center">)

[comment]: <> (<img src="FTN/results/noisy%20images%20batchsize16_lr_0.001_noise_0.2_layers_5.jpeg" alt="Noisy Images" width="70%"/>)

[comment]: <> (</p>)

[comment]: <> (Denoised Images During Training:)

[comment]: <> (<p align="center">)

[comment]: <> (<img src="FTN/results/denoising%20images%20batchsize16_lr_0.001_noise_0.2_layers_5.jpeg" alt="Denoised Images" width="70%"/>)

[comment]: <> (</p>)

Test Image on 0.2 std noise:
<p align="center">
<img src="FTN/results/clean_image.jpeg" alt="clean Images" width="50%"/>
</p>
<p align="center">
<img src="FTN/results/denoised_0.jpeg" alt="Denoised Images" width="50%"/>
</p>

<p align="center">
<img src="FTN/results/noisy_image.jpeg" alt="noisy Images" width="50%"/>
</p>

###Notes 
1. After 20 epochs the loss not converging (gets an error of 0.03 on L1 loss).
2. FTN layers initialized to identity (according to the paper).
3. The model has difficulty learning bigger noise  (#todo add an image of std 0.4 on 10 layers).


## 2nd Step

Finetune the model on Gaussian noise with std equal to [0.5, 0.6], alpha=1 while freezing
the weights of the model (beside the FTN layers, which responsible to learn the transition between the kernels on each noise),
At first the loss didn't converge, but after the initialization of the ftn kernel as idenety and the bias to zero the error has converged to a certain limit (which in my opinion is not good enough).

Results during Finetune the model on 0.5 std noise:

<p align="center">

<img src="readme_figures/finetune_0.5_7_layers.png" alt="Denoising" width="70%"/>

</p>

<p align="center">

<img src="readme_figures/finetune_0.5_7_layers_psnr.png" alt="PSNR" width="70%"/>

</p>

<p align="center">

<img src="readme_figures/finetune_0.5_7_layers_loss.png" alt="LOSS" width="70%"/>

</p>





### FTN- StyleTransfer Task
Because the noise cleaning model did not work adequately, it was difficult to see whether during interpolation in the model's parameters space, it clears noise with unlearned noise (level between the start level and the end one). Therefore, we chose to test the model on the style task (adding style to the image) - will allow us a clearer distinction.\\
At first I implemented the paper "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization", and tested it without adding the FTN blocks. 

 <p align="center">
  <img src="readme_figures/feathers_dan_first_step.png" alt="first step feathers style" width="45%"/>
  <img src="readme_figures/dan_second_mosaic_.png" alt="second step mosaic style" width="45%"/>
  </p>

# Related Work

## Representation Learning

### Filter Transition Network - Smoother Network Tuning and Interpolation for Continuous-level Image Processing

- [paper](https://arxiv.org/abs/2010.02270)
- no code


Filter Transition Network (FTN) which is a non-linear module that easily adapts to new levels (and a method to initialize non- linear CNNs with identity mappings). 

FTN takes CNN filters as input (instead of image/feature map) and learns the transitions between levels (the modification is non-linear which can better adapt to new level). The FTN layer basically transform the filters of the network. 



<p align="center">
<img src="readme_figures/ftn_flow.jpeg" alt="FTN Architecture" width="70%"/>
</p>

### AdaFM - Modulating Image Restoration with Continual Levels via Adaptive Feature Modification Layers (CVPR 2019)

- [paper](https://arxiv.org/abs/1904.08118)
- [code](https://github.com/hejingwenhejingwen/AdaFM) 


- Main Idea:
The additional module, namely AdaFM layer, performs channel-wise feature modification, and can adapt a model to another restoration level with high accuracy. the aim is to add another layer to manipulate the statistics of the filters, so that they could be adapted to another restoration level.

Adaptive Feature Modification (AdaFM) layers are inspired by the recent normalization methods in deep CNNs, but in contrast AdaFM layer is independent of either batch or instance samples, filter size and position of AdaFM layers are flexible, the interpolation property of AdaFM layers could achieve continual modulation of restoration levels.

Because there are “high correlation” between the filters of the networks (as we saw in DNI), we can think of a function that performs the match between them those filters. To further reveal their relationship, they use a filter to bridge the corresponding filters - g (done by depth-wise convolution layer after each layer). 

In addition, the filter g is gradually updated by gradient descent, so we can obtain the intermediate filter to get smooth transition.



<p align="center">
<img src="readme_figures/ada_flow.jpeg" alt="AdaFM Architecture" width="70%"/>
</p>

### DNI - Deep Network Interpolation for Continuous Imagery Effect Transition (CVPR 1028)

- [paper](https://arxiv.org/abs/1811.10515)
- [code](https://github.com/xinntao/DNI)


Continuous imagery effect transition is achieved via linear interpolation in the parameter space of existing trained networks. provided with a model for a particular effect A, they fine-tune it to realize another relevant effect B. DNI applies linear interpolation for all the corresponding parameters of these two deep networks (basically for the filters, normalization (IN) and the biases).

Network Interpolation:
If you train two models from scratch on the same task, and perform visualization of the filters (first and last filters), It can be seen that the order of the filters is different but their representation is similar (though not in the same order). Fine-tuning, however, can help to maintain the filters’s order and pattern. The “high correlation” between the parameters of these two networks provides the possibility for meaningful interpolation. 

In the image below it can be seen that the filters for different types of noise is strong correlated and when fine-tune several models for relevant tasks (different types of noise) from a pretrained one (N20), the correspondingly learned filters have intrinsic relations with a smooth transition, measuring the correlation between two filter (similar to the Pearson correlation) results close relationship among the learned filters, exhibiting a gradual change as the noise level increases. 


<p align="center">
<img src="readme_figures/filters_dni.jpeg" alt="DNI Filters" width="70%"/>
</p>

### Perceptual Losses for Real-Time Style Transfer and Super-Resolution

- [paper](https://arxiv.org/pdf/1603.08155.pdf)
- [code](https://github.com/dxyang/StyleTransfer)

In this paper they combined the benefits of a CNN for image transformation tasks (train in a supervised manner), 
and the benefit of perceptual loss functions based not on differences between pixels but instead on differences between high-level image feature representations extracted
from pretrained convolutional neural networks.

training feedforward transformation networks for image transformation tasks, but rather than
using per-pixel loss functions depending only on low-level pixel information, 
train the networks using perceptual loss functions that depend on high-level
features from a pretrained loss network. 



<p align="center">
<img src="readme_figures/filters_dni.jpeg" alt="DNI Filters" width="70%"/>
</p>


## Dynamic-Net: Tuning the Objective Without Re-training for Synthesis Tasks

- [paper](https://arxiv.org/pdf/1811.08760.pdf)
- [code](https://github.com/AlonShoshan10/dynamic_net)
 
we cannot directly modify the objective at test-time. However, what we can do is modify the
latent space representation. Therefore, their approach relies
on manipulation of deep features in order to emulate a manipulation in objective space.

The main advantages of the Dynamic-Net are three-fold.
1.Using a single training session the Dynamic-Net can
emulate networks trained with a variety of different objectives, for example, networks which produce stronger or
weaker stylization effects

2. The ability to traverse the objective space at test-time shrinks the search space during training

3. it facilitates image-specific and user-specific adaptation, without re-training.

<p align="center">
    <img src="readme_figures/dynamic.png" height="200px">
    <img src="readme_figures/dynamic1.png" height="200px">
    <img src="readme_figures/dynamic1.png" height="200px">
</p>
