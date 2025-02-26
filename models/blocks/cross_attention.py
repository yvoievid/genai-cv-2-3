from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim, emb_dim):
        super().__init__()
        self.query = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.key = nn.Linear(emb_dim, feature_dim)
        self.value = nn.Linear(emb_dim, feature_dim)
        self.scale = feature_dim ** -0.5
        self.proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
    
    def forward(self, x, emb):
        B, C, H, W = x.shape
        q = self.query(x).view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        k = self.key(emb).unsqueeze(1)  # (B, 1, C)
        v = self.value(emb).unsqueeze(1)  # (B, 1, C)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H*W, 1)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H*W, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        out = self.proj(out)
        return x + out
