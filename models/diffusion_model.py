import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class DiffusionModel(pl.LightningModule):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, lr=2e-4):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.lr = lr
        self.register_buffer('beta', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
    
    def forward_diffusion(self, x0, t, noise):
        # q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*noise
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    
    def training_step(self, batch, batch_idx):
        x, _ = batch  
        batch_size = x.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x.device)
        noise = torch.randn_like(x)
        x_noisy = self.forward_diffusion(x, t, noise)
        noise_pred = self.model(x_noisy, t)
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    @torch.no_grad()
    def sample(self, num_samples=16, ddim_steps=50):
        self.model.eval()
        device = next(self.model.parameters()).device
        img_size = 28
        x = torch.randn(num_samples, 1, img_size, img_size, device=device)
        ddim_timesteps = torch.linspace(self.timesteps - 1, 0, steps=ddim_steps, dtype=torch.long, device=device)
        
        for i in range(len(ddim_timesteps) - 1):
            t = ddim_timesteps[i]
            t_next = ddim_timesteps[i + 1]
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = self.model(x, t_batch)
            alpha_bar_t = self.alpha_bar[t]
            sqrt_alpha_bar_t = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
            alpha_bar_t_next = self.alpha_bar[t_next]
            sqrt_alpha_bar_t_next = alpha_bar_t_next.sqrt()
            x = sqrt_alpha_bar_t_next * x0_pred + (1 - alpha_bar_t_next).sqrt() * noise_pred
        return x
