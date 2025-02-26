import torch
from unet_with_deepcache import UNetWithDeepCache
import matplotlib.pyplot as plt

class RectifiedFlowDiffusionWithDeepCache(pl.LightningModule):
    def __init__(self, img_size=28, batch_size=64, lr=2e-4):
        super(RectifiedFlowDiffusionWithDeepCache, self).__init__()
        
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.model = UNetWithDeepCache(in_channels=1, out_channels=1)
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
    
    def generate_samples(self, num_samples=16, steps=100, enable_cache=False, cache_skip_steps=1):
        self.eval()
        self.model.enable_caching(enable_cache)
        
        # Start timer
        start_time = time.time()
        
        with torch.no_grad():
            # Start from random noise
            x = torch.randn(num_samples, 1, self.img_size, self.img_size, device=self.device)
            
            # Integrate velocity field
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.ones(num_samples, device=self.device) * (1.0 - i * dt)
                
                # When using DeepCache, decide whether to update cache
                if enable_cache and i % cache_skip_steps != 0 and i > 0:
                    # Reuse cached computations
                    pass
                else:
                    # Clear cache for the new step if we're updating
                    if enable_cache and i % cache_skip_steps == 0:
                        self.model.clear_cache()
                
                # Predict velocity field
                v = self.model(x, t)
                
                # Update x using Euler method
                x = x - v * dt
            
            # Clamp values to [0, 1]
            x = torch.clamp(x, 0, 1)
        
        # End timer and calculate
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Get cache statistics if enabled
        cache_stats = self.model.get_cache_hits() if enable_cache else {}
        
        # Disable caching after generation
        self.model.enable_caching(False)
        
        return x, generation_time, cache_stats
    
    def benchmark_generation(self, num_samples=16, steps=100, cache_skip_steps_list=[1, 2, 5, 10, 20]):
        """Benchmark generation with different cache skip steps"""
        results = []
        
        # Baseline: No caching
        samples_no_cache, time_no_cache, _ = self.generate_samples(
            num_samples=num_samples, 
            steps=steps, 
            enable_cache=False
        )
        results.append({
            'method': 'No Cache',
            'time': time_no_cache,
            'samples': samples_no_cache,
            'speedup': 1.0,
            'cache_hits': {}
        })
        
        for skip_steps in cache_skip_steps_list:
            samples, generation_time, cache_hits = self.generate_samples(
                num_samples=num_samples, 
                steps=steps, 
                enable_cache=True, 
                cache_skip_steps=skip_steps
            )
            
            similarity = F.mse_loss(samples, samples_no_cache).item()
            
            results.append({
                'method': f'DeepCache (skip={skip_steps})',
                'time': generation_time,
                'samples': samples,
                'speedup': time_no_cache / generation_time,
                'similarity': similarity,
                'cache_hits': cache_hits
            })
        
        return results
    
    def visualize_benchmark_results(self, benchmark_results, num_samples_to_show=4):
        """Visualize benchmark results"""
        # Plot samples
        fig, axes = plt.subplots(len(benchmark_results), num_samples_to_show, 
                                 figsize=(num_samples_to_show * 2, len(benchmark_results) * 2))
        
        for i, result in enumerate(benchmark_results):
            method = result['method']
            time_taken = result['time']
            speedup = result.get('speedup', 1.0)
            similarity = result.get('similarity', 'N/A')
            
            method_label = f"{method}\nTime: {time_taken:.2f}s, Speedup: {speedup:.2f}x"
            if similarity != 'N/A':
                method_label += f", Similarity: {similarity:.6f}"
            
            axes[i, 0].set_ylabel(method_label, fontsize=8)
            
            for j in range(num_samples_to_show):
                if j < result['samples'].shape[0]:
                    axes[i, j].imshow(result['samples'][j, 0].cpu().numpy(), cmap='gray')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.figure(figsize=(10, 6))
        methods = [result['method'] for result in benchmark_results]
        speedups = [result.get('speedup', 1.0) for result in benchmark_results]
        
        plt.bar(methods, speedups)
        plt.axhline(y=1.0, color='r', linestyle='-')
        plt.ylabel('Speedup Factor')
        plt.title('DeepCache Speedup Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        has_cache_stats = any(len(result.get('cache_hits', {})) > 0 for result in benchmark_results)
        if has_cache_stats:
            plt.figure(figsize=(12, 6))
            
            all_modules = set()
            for result in benchmark_results:
                all_modules.update(result.get('cache_hits', {}).keys())
            all_modules = sorted(all_modules)
            
            bar_width = 0.8 / (len(benchmark_results) - 1) if len(benchmark_results) > 1 else 0.4
            x = np.arange(len(all_modules))
            
            for i, result in enumerate(benchmark_results[1:], 1):  # Skip no-cache result
                hits = [result.get('cache_hits', {}).get(module, 0) for module in all_modules]
                plt.bar(x + (i-1) * bar_width - 0.4, hits, bar_width, label=result['method'])
            
            plt.xlabel('Model Components')
            plt.ylabel('Cache Hits')
            plt.title('DeepCache Hits by Model Component')
            plt.xticks(x, all_modules, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
        
        plt.show()