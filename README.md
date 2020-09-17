# acrc_footprints
The goal of this project is to develop an image-to-image translation solution to predict the shape of plumes based on metereological inputs.
This repo contains the necessary files to generate usable datasets and train and test them on the pix2pix model developed by junyanz (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

The repo is particularly oriented to be used in bluepebble (see the [step by step guide](step-by-step.md)). It requires Python 3, Pytorch, CPU or NVIDIA GPU + CUDA (all these already available in bluepebble).

**To Do**
- [x] Merge all data_generation files into one and add main arguments
- [ ] Add secondary arguments 
    - [ ] Different cut sizes for input and output
    - [ ] More input options 
- [ ] Add sample databases, checkpoints and results
    - [x] Add sample databases
    - [x] Add sample checkpoints and results
- [ ] Develop, improve and add arguments to loss plotting and results plotting scripts
    - [ ]  Include updates in step_by_step 
- [x] Add new document with
    - [x] Summary of tested models
    - [x] Suggestions to go forward

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

The scripts will work with any Python 3.7 module or above. You can load them on bluepebble using
```
module load lang/python/anaconda/3.7-2019.10
```
Similarly, you can load the pytorch module using
```
module load lang/python/anaconda/pytorch
```
Only one of them can be activated at a given time, so you might need to load one or the other depending on the stage.

# Contents
This repo contains the following:
- data_generation: scripts on how to produce appropriate datasets
- databases: sample databases
- checkpoints: checkpoints of previously trained models (trained on the sample databases), plus a plotting tool to visualise the loss graphs.
- results: the results of previously trained models (trained on the sample databases), plus a plotting tool to visualise and compare the results.

All folders and steps are described in the [step by step guide](step-by-step.md).
# Some notes on the pix2pix algorithm
See their [paper](https://arxiv.org/pdf/1611.07004.pdf) for some information.
#### Loss function
 pix2pix's loss function is cGAN_loss + lambda*L1_loss. 
- conditional GANs are adversarial networks: there is a generator that tries to produce realistic images given a certain input, and a discriminator that decides if the image has been generated or has been extracted from the original dataset. The generator tries to minimise the loss, where the discriminator tries to maximise it. Given this adversarial nature, it is common that the cGAN loss never converges.
- L1_loss, also called Least Absolute Deviations, is the sum of (Ytrue - Y predicted)). It produces accurate (spatially correct) images, but tends to blur edges and thus produce "non-realistic looking" images.
- lambda controls the importance of L1 vs cGAN - lower values of lambda will produce more correct-looking but inaccurate plumes, whereas higher values will produce plumes in the correct location with blurred edges and colour. The default value in pix2pix is 100.
