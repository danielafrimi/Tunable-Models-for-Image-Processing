# Lab

* Train a renset model (with 5-10 layers) on denoising task.
added a gausian noise with std equal to 0.2 to each image. 
I've added the ftn layer after each filter, In the first training alpha=0 (we dont learn the ftn blocks beacuse the gradient is 0).
the results were the same to the basic model (PSNR 27-28).

Clean Images During Training:

<p align="center">
<img src="FTN/results/clean%20images%20batchsize16_lr_0.001_noise_0.2_layers_5.jpeg" alt="Clean Images" width="70%"/>
</p>

Noisy Images During Training:

Denoising Images During Training:


Test Image:

Notes: 
1. After 20 epochs the loss not converging (gets an error of 0.03 on L1 loss).
2. FTN layers initialized to identity (according to the paper).


* 

# Related Work

## Representation Learning

### SimCLR - A Simple Framework for Contrastive Learning of Visual Representations (Feb 2020)

- Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton.  
  Google Research, Brain Team.
- Accepted to ICML 2020.
- [paper](https://arxiv.org/pdf/2002.05709.pdf)
- [code](https://github.com/google-research/simclr)

This paper presents SimCLR: A simple framework for contrastive learning of visual representations. \
The self-supervised task is to identify that different augmentations of the same image are the same.

<p align="center">
<img src="images/simclr_architecture.png" alt="SimCLR Architecture" width="70%"/>
</p>





1.  הfinetune להראות החלקה של 
2. להראות החלקה של תמונה עם רעש 0.2
3. להראות את הloss בזמן הfinetune
4.  ולהראות תוצאות לחשב אינטפולציה
5. fix psnr
