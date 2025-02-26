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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


pl.seed_everything(42)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256, hidden_dims=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.time_dim = time_dim
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.initial_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        in_channels = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(),
                    nn.MaxPool2d(2)
                )
            )
            in_channels = hidden_dim
        
        self.middle_block = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.SiLU(),
        )
        
        self.time_embeddings = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.time_embeddings.append(
                nn.Sequential(
                    nn.Linear(time_dim, hidden_dim),
                    nn.SiLU()
                )
            )
        
        self.up_blocks = nn.ModuleList()
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_hidden_dims) - 1):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_hidden_dims[i], reversed_hidden_dims[i+1], kernel_size=2, stride=2),
                    nn.Conv2d(reversed_hidden_dims[i+1] * 2, reversed_hidden_dims[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(reversed_hidden_dims[i+1]),
                    nn.SiLU(),
                    nn.Conv2d(reversed_hidden_dims[i+1], reversed_hidden_dims[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(reversed_hidden_dims[i+1]),
                    nn.SiLU(),
                )
            )
        
        self.final_conv = nn.Conv2d(hidden_dims[0], out_channels, kernel_size=1)
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        time_emb = self.time_mlp(t)
        x = self.initial_conv(x)
        skips = [x]
        
        x = x + self.time_embeddings[0](time_emb).unsqueeze(-1).unsqueeze(-1)
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            x = x + self.time_embeddings[i+1](time_emb).unsqueeze(-1).unsqueeze(-1)
            skips.append(x)
        
        x = self.middle_block(x)
        skips = skips[:-1]  
        skips.reverse()
        
        for i, block in enumerate(self.up_blocks):
            x = block[0](x)
            if x.shape != skips[i].shape:
                x = F.interpolate(x, size=skips[i].shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skips[i]], dim=1)
            for j in range(1, len(block)):
                x = block[j](x)

        return self.final_conv(x)
    
    
    