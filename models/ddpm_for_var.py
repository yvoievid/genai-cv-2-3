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
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

import pytorch_lightning as pl

class DDPM(pl.LightningModule):
    def __init__(self, vae, latent_dim=20, num_timesteps=1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.vae = vae
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, z_t, t):
        return self.model(z_t)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        with torch.no_grad():
            _, mu, _ = self.vae(x)  # Encode images via pretrained VAE
        
        t = torch.randint(0, self.num_timesteps, (x.size(0),), device=self.device).long()
        noise = torch.randn_like(mu)
        noisy_mu = mu + noise
        noise_pred = self(noisy_mu, t)
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


