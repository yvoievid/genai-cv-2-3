
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from models.blocks.rectified_decoder import Decoder
from models.blocks.rectified_encoder import Encoder
from unet import UNet
import torchvision

class RectifiedFlowDiffusion(pl.LightningModule):
    def __init__(self, img_size=28, batch_size=64, lr=2e-4):
        super(RectifiedFlowDiffusion, self).__init__()
        
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.model = UNet(in_channels=1, out_channels=1)
        self.save_hyperparameters()
        
    def forward(self, x, t):
        return self.model(x, t)
    
    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        t = torch.rand(imgs.shape[0], device=self.device)
        z = torch.randn_like(imgs)
        x_t = (1 - t.view(-1, 1, 1, 1)) * imgs + t.view(-1, 1, 1, 1) * z
        v_true = z - imgs  
        v_pred = self.model(x_t, t)
        loss = F.mse_loss(v_pred, v_true)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        t = torch.rand(imgs.shape[0], device=self.device)

        z = torch.randn_like(imgs)

        x_t = (1 - t.view(-1, 1, 1, 1)) * imgs + t.view(-1, 1, 1, 1) * z

        v_true = z - imgs 
        
        v_pred = self.model(x_t, t)
        loss = F.mse_loss(v_pred, v_true)
        
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def generate_samples(self, num_samples=16, steps=100):
        """Generate samples using numerical integration of the predicted velocity field"""
        self.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, 1, self.img_size, self.img_size, device=self.device)
            
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.ones(num_samples, device=self.device) * (1.0 - i * dt)
                v = self.model(x, t)
                x = x - v * dt

            x = torch.clamp(x, 0, 1)
        
        return x
    
    def on_train_epoch_end(self):
        if self.current_epoch % 5 == 0:
            samples = self.generate_samples(num_samples=16)
            grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)

class LatentRectifiedFlowDiffusion(pl.LightningModule):
    def __init__(self, latent_dim=64, img_size=28, batch_size=64, lr=2e-4, kl_weight=0.0001):
        super(LatentRectifiedFlowDiffusion, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.kl_weight = kl_weight
        
        self.encoder = Encoder(in_channels=1, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=1, img_size=img_size)
        
        self.diffusion = LatentMLP(latent_dim=latent_dim)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        z, _, _ = self.encoder(x)
        return self.decoder(z)
    
    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        z, mu, log_var = self.encoder(imgs)
        t = torch.rand(imgs.shape[0], device=self.device)
        noise = torch.randn_like(z)
        z_t = (1 - t.view(-1, 1)) * z + t.view(-1, 1) * noise
        v_true = noise - z  
        v_pred = self.diffusion(z_t, t)
        diffusion_loss = F.mse_loss(v_pred, v_true)
        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x_recon, imgs)
        
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)
        loss = diffusion_loss + recon_loss + self.kl_weight * kl_loss
        
        self.log('train_loss', loss)
        self.log('train_diffusion_loss', diffusion_loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_kl_loss', kl_loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        z, mu, log_var = self.encoder(imgs)
        t = torch.rand(imgs.shape[0], device=self.device)
        noise = torch.randn_like(z)
        z_t = (1 - t.view(-1, 1)) * z + t.view(-1, 1) * noise
        v_true = noise - z
        v_pred = self.diffusion(z_t, t)
        diffusion_loss = F.mse_loss(v_pred, v_true)
        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x_recon, imgs)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)
        loss = diffusion_loss + recon_loss + self.kl_weight * kl_loss
        
        self.log('val_loss', loss)
        self.log('val_diffusion_loss', diffusion_loss)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kl_loss', kl_loss)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def generate_samples(self, num_samples=16, steps=100):
        """Generate samples using numerical integration of the predicted velocity field in latent space"""
        self.eval()
        with torch.no_grad():
            latent_noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.ones(num_samples, device=self.device) * (1.0 - i * dt)
                
                v = self.diffusion(latent_noise, t)
                
                latent_noise = latent_noise - v * dt
            
            samples = self.decoder(latent_noise)
        
        return samples
    
    def on_train_epoch_end(self):
        if self.current_epoch % 5 == 0:
            samples = self.generate_samples(num_samples=16)
            grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
