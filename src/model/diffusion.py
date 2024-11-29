#!/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from . import unet
from pytorch_wavelets import DWTForward, DWTInverse
from torch.nn import functional as F


class DWSR(nn.Module):
    def __init__(self, image_size):
        super.__init__()
        self.image_size = image_size

    def forward(self, x):
        return 0  # TODO fix this


class DDPM(nn.Module):
    def __init__(
        self,
        input_channels,
        image_size,
        num_timesteps=1000,
        scale_factor=2,
        device="cuda",
    ):
        self.noise_predictor = unet.UNet(
            image_size=image_size, in_channels=input_channels
        ).to(device)
        self.dwsr = DWSR(image_size=image_size).to(device)
        self.optimiser = optim.Adam([self.noise_predictor.parameters(), self.dwsr.parameters()])
        self.loss = nn.MSELoss()
        self.num_timesteps = num_timesteps
        self.define_noise_schedule()
        self.downres = v2.Compose([v2.Resize(image_size // scale_factor), v2.Resize(image_size)])
        self.dwt = DWTForward().to(device)
        self.idwt = DWTInverse().to(device)

    def define_noise_schedule(self, mode="linear"):
        if mode != "linear":
            raise NotImplementedError(mode)
        betas = torch.linspace(1e-6, 1e-2, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        recip_sqrt_alphas = 1.0 / torch.sqrt(alphas)
        recip_sqrt_1m_alphas_cumprod = 1.0 / torch.sqrt(1.0 - alphas_cumprod)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_1m_alphas_cumprod", sqrt_1m_alphas_cumprod)
        self.register_buffer("recip_sqrt_alphas", recip_sqrt_alphas)
        self.register_buffer(
            "recip_sqrt_1m_alphas_cumprod", recip_sqrt_1m_alphas_cumprod
        )

    def apply_dwt(self, x):
        ll, [(lh, hl, hh)] = self.dwt(x)
        return torch.stack(
            [ll, lh[:, :, 0, :, :], hl[:, :, 1, :, :], hh[:, :, 2, :, :]], dim=1
        )

    def apply_idwt(self, x, target_width, target_height):
        sr_images_LL = x[:, 0:3, :, :]
        sr_images_HL = x[:, 3:6, :, :].unsqueeze(2)
        sr_images_LH = x[:, 6:9, :, :].unsqueeze(2)
        sr_images_HH = x[:, 9:12, :, :].unsqueeze(2)

        sr_HFreqs = torch.cat([sr_images_HL, sr_images_LH, sr_images_HH], 2)
        sr_images = self.ifm((sr_images_LL, [sr_HFreqs]))
        sr_images = F.interpolate(
            sr_images, size=(target_width, target_height), mode="bicubic"
        )

        return sr_images

    def apply_noise(self, x, t):
        noise = torch.rand_like(x)
        sqrt_gam_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_gam_t = self.sqrt_1m_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_gam_t * x + sqrt_1m_gam_t * noise, noise

    def forward(self, x):
        """
        Predict noise
        """
        x_hr = x
        x_lr = self.downres(x)

        x_hr = self.apply_dwt(x_hr)
        x_lr = self.apply_dwt(x_lr)

        x_pred = self.dwsr(x)

        x_residual = x_hr - x_pred

        t = torch.randint(1, self.num_timesteps + 1)
        x_noisy = self.apply_noise(torch.cat([x_lr, x_residual], dim=1), t)

        pred_noise, real_noise = self.noise_predictor(x_noisy, t)
        loss = self.loss(real_noise, pred_noise)
        loss.backward()
        self.optimiser.step()
        return loss.item()
