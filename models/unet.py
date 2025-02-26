from blocks.decoder import DecoderBlock
from blocks.encoder import EncoderBlock
from blocks.zero_conv import ZeroConv
from torch import nn
import torch

class UNetForControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = EncoderBlock(1, 32, use_bn=False)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        
        self.bridge = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.dec1 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec3 = DecoderBlock(64, 32)
        
        self.output = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x, control_features=None):
        p1, s1 = self.enc1(x)
        p2, s2 = self.enc2(p1)
        p3, s3 = self.enc3(p2)
        
        b = self.bridge(p3)
        
        if control_features is not None:
            c_s1, c_s2, c_s3, c_b = control_features
            s1 = s1 + c_s1
            s2 = s2 + c_s2
            s3 = s3 + c_s3
            b = b + c_b
         
        d1 = self.dec1(b, s3)
        d2 = self.dec2(d1, s2)
        d3 = self.dec3(d2, s1)
        
        return torch.sigmoid(self.output(d3))
