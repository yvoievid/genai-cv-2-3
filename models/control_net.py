from blocks.decoder import DecoderBlock
from blocks.encoder import EncoderBlock
from blocks.zero_conv import ZeroConv
from torch import nn

class ControlNet(nn.Module):
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

        self.zero_s1 = ZeroConv(32, 32)
        self.zero_s2 = ZeroConv(64, 64)
        self.zero_s3 = ZeroConv(128, 128)
        self.zero_b = ZeroConv(256, 256)
        
    def forward(self, control):
        p1, s1 = self.enc1(control)
        p2, s2 = self.enc2(p1)
        p3, s3 = self.enc3(p2)
        b = self.bridge(p3)
        c_s1 = self.zero_s1(s1)
        c_s2 = self.zero_s2(s2)
        c_s3 = self.zero_s3(s3)
        c_b = self.zero_b(b)
        
        return (c_s1, c_s2, c_s3, c_b)