#!/bin/bash

import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms import v2
import .unet

class DWSR(nn.Module):
    def __init__(self, image_size):
        super.__init__()
        self.image_size = image_size

    def forward(self, x):
        return 0 # TODO fix this

class DDPM(nn.Module):
    def __init__(self, input_channels, image_size, num_timesteps=1000, scale_factor=2):
        self.noise_predictor = unet.UNet(image_size=image_size, in_channels=input_channels)
        self.dwsr = DWSR(image_size=image_size)
        self.optimiser = optim.Adam(noise_predictor.params)
        self.num_timesteps = num_timesteps 
        define_noise_schedule()
        self.downres = v2.Compose([v2.Resize(image_size//2), v2.Resize(image_size)])
    

    def define_noise_schedule(self, mode='linear'):
        if mode != 'linear':
            raise NotImplementedError(mode)
        betas = torch.linspace(1e-6, 1e-2, self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        recip_sqrt_alphas = 1. / torch.sqrt(alphas)
        1m_alphas = 1. - alphas
        recip_sqrt_1m_alphas_cumprod = 1. / torch.sqrt(1. - alphas_cumprod)

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_1m_alphas_cumprod', sqrt_1m_alphas_cumprod)
        self.register_buffer('recip_sqrt_alphas', recip_sqrt_alphas)
        self.register_buffer('recip_sqrt_1m_alphas_cumprod', recip_sqrt_1m_alphas_cumprod)
        
        
    def forward(self, batch):
        batch_hr = batch
        batch_lr = self.downres(batch_hr)
        sr_pred = self.dwsr(sr_pred)
        
