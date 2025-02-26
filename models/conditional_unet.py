from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from blocks.downsample import DownSample
from blocks.uncond import UncondBlock
from blocks.sinusoidal_pos import SinusoidalPosEmb
from blocks.cross_attention import CrossAttentionBlock
from blocks.upsample import UpSample


class CombinedConditioningUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128,
                 num_classes=10, class_emb_dim=32, attn_emb_dim=64):
        """
        in_channels: image channels (e.g., 1 for MNIST)
        base_channels: base channel count for the U-Net
        time_emb_dim: dimensionality for time embedding
        num_classes: number of classes in the dataset
        class_emb_dim: dimension for the input channel conditioning embedding
        attn_emb_dim: dimension for the cross-attention embedding
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_emb_input = nn.Embedding(num_classes, class_emb_dim)
        self.class_emb_attn = nn.Embedding(num_classes, attn_emb_dim)
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)
        
        self.conv_in = nn.Conv2d(in_channels + class_emb_dim, base_channels, kernel_size=3, padding=1)
        
        self.block1 = UncondBlock(base_channels, base_channels, time_emb_dim)
        self.down1 = DownSample(base_channels, base_channels * 2)
        self.block2 = UncondBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.down2 = DownSample(base_channels * 2, base_channels * 4)
        self.block3 = UncondBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        self.bottleneck = UncondBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.cross_attn = CrossAttentionBlock(feature_dim=base_channels * 4, emb_dim=attn_emb_dim)
        
        self.up1 = UpSample(base_channels * 4, base_channels * 2)
        self.block4 = UncondBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.up2 = UpSample(base_channels * 2, base_channels)
        self.block5 = UncondBlock(base_channels, base_channels, time_emb_dim)
        
        self.conv_out = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t, y):
        """
        x: (B, 1, H, W) noisy image.
        t: (B,) timesteps.
        y: (B,) class labels. If y < 0 the condition is dropped.
        """
        B, _, H, W = x.shape
        mask = (y >= 0).float().unsqueeze(1)
        y_idx = y.clone()
        y_idx[y_idx < 0] = 0 
        class_emb_in = self.class_emb_input(y_idx) * mask 
        class_emb_in = class_emb_in.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        x = torch.cat([x, class_emb_in], dim=1)
        
        h = self.conv_in(x)
        t_emb = self.time_embedding(t)  
        h = self.block1(h, t_emb)
        skip1 = h
        h = self.down1(h)
        h = self.block2(h, t_emb)
        skip2 = h
        h = self.down2(h)
        h = self.block3(h, t_emb)
        

        h = self.bottleneck(h, t_emb)

        mask_attn = (y >= 0).float().unsqueeze(1)
        y_idx_attn = y.clone()
        y_idx_attn[y_idx_attn < 0] = 0
        class_emb_attn = self.class_emb_attn(y_idx_attn) * mask_attn
        h = self.cross_attn(h, class_emb_attn)

        h = self.up1(h)
        h = h + skip2
        h = self.block4(h, t_emb)
        h = self.up2(h)
        h = h + skip1
        h = self.block5(h, t_emb)
        return self.conv_out(h)
