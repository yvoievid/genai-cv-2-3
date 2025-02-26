
from torch import nn
import torch

class UNetWithDeepCache(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256, hidden_dims=[32, 64, 128, 256]):
        super(UNetWithDeepCache, self).__init__()
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
        
        self.cache = {}
        self.cache_enabled = False
        self.cache_hit_counter = defaultdict(int)
    
    def enable_caching(self, enabled=True):
        """Enable or disable the DeepCache mechanism"""
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()
        self.cache_hit_counter = defaultdict(int)
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}
    
    def get_cache_hits(self):
        """Return the cache hit statistics"""
        return dict(self.cache_hit_counter)
    
    def cache_key(self, t, step_idx):
        """Generate a unique key for caching based on timestep and step index"""
        return f"{float(t):.6f}_{step_idx}"
    
    def forward(self, x, t):
        batch_size = x.shape[0]
        
        t_float = t.clone().detach()
        t = t.unsqueeze(-1)
        time_emb = self.time_mlp(t)
        
        x = self.initial_conv(x)
        
        skips = [x]
        
        x = x + self.time_embeddings[0](time_emb).unsqueeze(-1).unsqueeze(-1)
        
        for i, block in enumerate(self.down_blocks):
            if self.cache_enabled and not self.training:
                cache_key = self.cache_key(t_float[0], f"down_{i}")
                if cache_key in self.cache:
                    x = self.cache[cache_key].clone()
                    self.cache_hit_counter[f"down_{i}"] += 1
                    skips.append(x)
                    continue
            
            x = block(x)
            x = x + self.time_embeddings[i+1](time_emb).unsqueeze(-1).unsqueeze(-1)
            skips.append(x)
            
            if self.cache_enabled and not self.training:
                self.cache[self.cache_key(t_float[0], f"down_{i}")] = x.clone()
        
        if self.cache_enabled and not self.training:
            cache_key = self.cache_key(t_float[0], "middle")
            if cache_key in self.cache:
                x = self.cache[cache_key].clone()
                self.cache_hit_counter["middle"] += 1
            else:
                x = self.middle_block(x)
                self.cache[cache_key] = x.clone()
        else:
            x = self.middle_block(x)
        
        skips = skips[:-1]
        skips.reverse()
        
        for i, block in enumerate(self.up_blocks):
            if self.cache_enabled and not self.training:
                cache_key = self.cache_key(t_float[0], f"up_{i}")
                if cache_key in self.cache:
                    x = self.cache[cache_key].clone()
                    self.cache_hit_counter[f"up_{i}"] += 1
                    continue
            
            x = block[0](x)
            
            if x.shape != skips[i].shape:
                # Resize if necessary
                x = F.interpolate(x, size=skips[i].shape[2:], mode='bilinear', align_corners=False)
                
            x = torch.cat([x, skips[i]], dim=1)

            for j in range(1, len(block)):
                x = block[j](x)
            

            if self.cache_enabled and not self.training:
                self.cache[cache_key] = x.clone()
        
        return self.final_conv(x)
