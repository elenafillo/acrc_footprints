# acrc_footprints
The goal of this project is to develop an image-to-image translation solution to predict the shape of plumes based on metereological inputs.
This repo contains the necessary files to generate usable datasets and train and test them on the pix2pix model developed by junyanz (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

The repo is particularly oriented to be used in bluepebble (see the [step by step guide](step-by-step.md)). It requires Python 3, Pytorch, CPU or NVIDIA GPU + CUDA (all these already available in bluepebble).

# Installation
Clone this repo 
```
git clone https://github.com/elenafillo/acrc_footprints
```
and clone the pix2pix repo
```
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
```
Install the pix2pix requirements (see their documentation) or if in bluepebble use 
```
cd acrc_footprints
pip install -r utils/requirements_bluepebble.txt
```
This avoid overlaps with the modules already installed in the bluepebble system.

# Contents
This repo contains the following:
- data_generation: scripts on how to produce appropriate datasets
- databases: sample databases
- checkpoints: checkpoints of previously trained models (trained on the sample databases), plus a plotting tool to visualise the loss graphs.
- results: the results of previously trained models (trained on the sample databases), plus a plotting tool to visualise and compare the results.

# Some notes on the pix2pix algorithm
See their [paper](https://arxiv.org/pdf/1611.07004.pdf) for some information.
#### Loss function
 pix2pix's loss function is cGAN_loss + lambda*L1_loss. 
- conditional GANs are adversarial networks: there is a generator that tries to produce realistic images given a certain input, and a discriminator that decides if the image has been generated or has been extracted from the original dataset. The generator tries to minimise the loss, where the discriminator tries to maximise it. Given this adversarial nature, it is common that the cGAN loss never converges.
- L1_loss, also called Least Absolute Deviations, is the sum of (Ytrue - Y predicted)). It produces accurate (spatially correct) images, but tends to blur edges and thus produce "non-realistic looking" images.
- lambda controls the importance of L1 vs cGAN - lower values of lambda will produce more correct-looking but inaccurate plumes, whereas higher values will produce plumes in the correct location with blurred edges and colour. The default value in pix2pix is 100.
