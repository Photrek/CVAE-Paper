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
    
    # Added clamp to prevent inf bugs!
    U1 = torch.rand(samples).clamp(min=1e-7) 
    U2 = F.normalize(torch.randn(samples, dim), dim=1) 
    
    R_sq = coupled_log(U1.pow(-2), kappa) 
    R = torch.sqrt(F.relu(R_sq)) 
    
    return R.unsqueeze(1) * U2 

# --- EXPERIMENT SETTINGS ---
base_samples = 100000
latent_dim = 10
kappa_original = 1.0

# IE parameters
kappa_IE = kappa_original / (1.0 + kappa_original)

# --- TOGGLE YOUR SCALE FACTOR HERE ---
# scale_IE_factor = 1.0 / np.sqrt(1.0 + kappa_original)  # The mathematically correct scale
scale_IE_factor = (1.0 + 2*kappa_original)**2            # The inflated scale

print(f"--- Multi-Sample IE Experiment (d={latent_dim}, base_samples={base_samples}) ---")
print(f"Original Kappa: {kappa_original}")
print(f"IE Kappa: {kappa_IE:.4f} | IE Scale Factor: {scale_IE_factor:.4f}\n")

# 1. Original Distribution (1 sample per input)
orig_samples = sample_coupled_gaussian(base_samples, latent_dim, kappa_original)
max_orig = torch.max(torch.abs(orig_samples)).item()

# Estimate 2nd moment (variance) per dimension
var_orig = torch.mean(orig_samples ** 2).item() 

print(f"Max Original (1 sample):  {max_orig:.2f}")
print(f"2nd Moment (Original):    {var_orig:.2f}  <-- (This will bounce wildly between runs!)\n")

# 2. IE Distribution with increasing num_samples
for num_samples in [1, 10, 50, 100, 200, 300]:
    total_ie_samples = base_samples * num_samples
    ie_samples = sample_coupled_gaussian(total_ie_samples, latent_dim, kappa_IE) * scale_IE_factor
    
    max_ie = torch.max(torch.abs(ie_samples)).item()
    var_ie = torch.mean(ie_samples ** 2).item() # Estimate 2nd moment per dimension
    
    print(f"Indep-Equals ({num_samples:3} samples) | Max: {max_ie:8.2f} (Ratio: {max_orig/max_ie:6.2f}x) | 2nd Moment: {var_ie:8.2f}")