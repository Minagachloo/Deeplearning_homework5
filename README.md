# Deeplearning_homework5

# DDPM on CIFAR-10

Implementation of Denoising Diffusion Probabilistic Model (DDPM) for image generation on the CIFAR-10 dataset.

## Overview

This project trains a DDPM model to generate 32×32 images. DDPM works by learning to reverse a gradual noising process - starting from random noise and iteratively denoising to produce realistic images.

## Requirements

- Python 3
- PyTorch
- denoising-diffusion-pytorch
```bash
