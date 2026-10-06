import torch
import torch.nn.functional as F
import numpy as np

def coupled_log(x, kappa, eps=1e-38):
    if abs(kappa) < 1e-12:
        return torch.log(torch.clamp(x, min=eps))
    x_safe = torch.clamp(x, min=eps)
    return (x_safe.pow(kappa) - 1.0) / kappa

def sample_coupled_gaussian(samples, dim, kappa):
    if kappa == 0:
        return torch.randn(samples, dim)
    
    # The degrees of freedom for the Coupled Gaussian
    nu = 2.0 / kappa
    
    # Draw scale mixture weights from a Gamma distribution
    gamma_dist = torch.distributions.Gamma(torch.tensor([nu / 2.0]), torch.tensor([nu / 2.0]))
    v = gamma_dist.sample((samples,)).squeeze(-1)
    
    # Prevent division by zero for extreme kappas
    v = torch.clamp(v, min=1e-7)
    
    # Draw standard N-dimensional normal samples
    z = torch.randn(samples, dim)
    
    # Apply the scale mixture (this perfectly shapes the heavy tails for ANY dimension!)
    return z / torch.sqrt(v).unsqueeze(1)

# --- EXPERIMENT SETTINGS ---
base_samples = 100000
latent_dim = 10
kappa_original = 1.0

# IE parameters
kappa_IE = kappa_original / (1.0 + kappa_original)

scale_IE_factor = 1.0 / np.sqrt(1.0 + kappa_original)  
        

print(f"--- Multi-Sample IE Experiment (d={latent_dim}, base_samples={base_samples}) ---")
print(f"Original Kappa: {kappa_original}")
print(f"IE Kappa: {kappa_IE:.4f} | IE Scale Factor: {scale_IE_factor:.4f}\n")

# 1. Original Distribution (1 sample per input)
orig_samples = sample_coupled_gaussian(base_samples, latent_dim, kappa_original)
max_orig = torch.max(torch.abs(orig_samples)).item()

# Estimate 2nd moment (variance) per dimension
var_orig = torch.mean(orig_samples ** 2).item() 

print(f"Max Original (1 sample):  {max_orig:.2f}")
print(f"2nd Moment (Original):    {var_orig:.2f} \n")

# 2. IE Distribution with increasing num_samples
for num_samples in [1, 10, 50, 100]:
    total_ie_samples = base_samples * num_samples
    ie_samples = sample_coupled_gaussian(total_ie_samples, latent_dim, kappa_IE) * scale_IE_factor
    
    max_ie = torch.max(torch.abs(ie_samples)).item()
    var_ie = torch.mean(ie_samples ** 2).item() # Estimate 2nd moment per dimension
    
    print(f"Indep-Equals ({num_samples:3} samples) | Max: {max_ie:8.2f} (Ratio: {max_orig/max_ie:6.2f}x) | 2nd Moment: {var_ie:8.2f}")