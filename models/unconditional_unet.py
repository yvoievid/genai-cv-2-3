
from models.blocks.sinusoidal_pos import SinusoidalPosEmb
from models.blocks.uncond import UncondBlock
from models.blocks.downsample import DownSample
from models.blocks.upsample import UpSample
from skimage.metrics import structural_similarity as ssim

import torch.nn as nn
import torch.nn.functional as F



class UnconditionalUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.block1 = UncondBlock(base_channels, base_channels, time_emb_dim)
        self.down1 = DownSample(base_channels, base_channels * 2)
        self.block2 = UncondBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.down2 = DownSample(base_channels * 2, base_channels * 4)
        self.block3 = UncondBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        self.bottleneck = UncondBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        self.up1 = UpSample(base_channels * 4, base_channels * 2)
        self.block4 = UncondBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.up2 = UpSample(base_channels * 2, base_channels)
        self.block5 = UncondBlock(base_channels, base_channels, time_emb_dim)
        
        self.conv_out = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        t_emb = self.time_embedding(t) 
        
        h = self.conv_in(x)
        
        h = self.block1(h, t_emb)
        skip1 = h
        h = self.down1(h)
        h = self.block2(h, t_emb)
        skip2 = h
        h = self.down2(h)
        h = self.block3(h, t_emb)
        
        h = self.bottleneck(h, t_emb)
        
        h = self.up1(h)
        h = h + skip2
        h = self.block4(h, t_emb)
        h = self.up2(h)
        h = h + skip1
        h = self.block5(h, t_emb)
        return self.conv_out(h)
    