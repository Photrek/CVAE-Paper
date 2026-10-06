import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import builtins


def dummy_input_generator():
    yield "0.0"  # Dummy kappa
    yield "100"  # Dummy dim
    yield "1"    # Dummy samples
    yield "3"    # Dummy mode (Safely skips dataset extraction!)
    while True:
        yield "3"

dummy_gen = dummy_input_generator()
original_input = builtins.input
builtins.input = lambda _: next(dummy_gen)

from run import CelebAImageDataset, CoupledVariationalAutoencoder, compute_normalization_constant_term
builtins.input = original_input

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.abspath('.')
DATA_DIR = '/mnt/data/RiskIntel/cvae-paper/celeba/celeba/img_align_celeba/'
OUTPUT_FILE = "additivity_experiment_results.txt"

BATCH_SIZE = 32
NUM_SAMPLES = 5
LATENT_DIM = 100
PRIOR_MEAN = 0.0
PRIOR_VAR = 1.0
NUM_TEST_IMAGES = 1000

# ==========================================
# DATASET INITIALIZATION
# ==========================================
def load_quick_testing_dataset():
    eval_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    full_dataset = CelebAImageDataset(directory_path=DATA_DIR, transform_pipeline=eval_transforms)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    
    torch.manual_seed(42)
    indices = torch.randperm(len(full_dataset)).tolist()
    testing_start = train_size + val_size
    testing_indices = indices[testing_start : testing_start + NUM_TEST_IMAGES]
    
    return DataLoader(Subset(full_dataset, testing_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================================
# KAPPA ADDITIVITY LOGIC
# ==========================================
def evaluate_pseudo_additivity(model, dataloader, kappa, a_xz):
    """
    Method 1: Standard Sum -> FE_1 + FE_2 + ... + FE_10
    Method 2: Kappa Sum -> FE_1 ⊕ FE_2 ⊕ ... ⊕ FE_10
              where A ⊕ B = A + B + kappa * A * B
    """
    model.eval()
    total_m1, total_m2 = 0.0, 0.0
    total_images = 0
    term_a_prior = compute_normalization_constant_term(kappa, LATENT_DIM).to(DEVICE)
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=f"Evaluating Kappa {kappa}", leave=False):
            images = images.to(DEVICE)
            current_batch_size = images.size(0)
            
            latent_mean, latent_logvar = model.encode(images)
            latent_z = model.apply_reparameterization_trick(
                latent_mean, latent_logvar, kappa, LATENT_DIM, scale_covariance=True, num_samples=NUM_SAMPLES
            )
            
            decoded = model.decode(latent_z)
            expanded_images = images.unsqueeze(1).expand(-1, NUM_SAMPLES, -1, -1, -1)
            
            # Reconstruction Loss for each of the 10 samples
            recon_loss = F.mse_loss(decoded, expanded_images, reduction='none')
            recon_loss = recon_loss.view(current_batch_size, NUM_SAMPLES, -1).sum(dim=2) 
            
            # Analytical KLD for the image
            trace_term = torch.exp(latent_logvar).sum(dim=1)
            mahalanobis = (latent_mean ** 2).sum(dim=1)
            comp_1 = -0.5 * LATENT_DIM * (1.0 + kappa * a_xz.float())
            comp_2 = 0.5 * (1.0 + kappa * term_a_prior.float()) * (mahalanobis + trace_term)
            comp_3 = -0.5 * a_xz.float() + 0.5 * term_a_prior.float()
            kld_loss = comp_1 + comp_2 + comp_3
            
            # Free Energy for each sample [batch_size, 10]
            free_energies = recon_loss + kld_loss.unsqueeze(1) 
            
            # METHOD 1: Standard Additivity (Sum)
            m1_batch = free_energies.sum(dim=1)
            
            # METHOD 2: Tsallis Pseudo-Additivity (Kappa-Sum)
            m2_batch = free_energies[:, 0].clone()
            for s in range(1, NUM_SAMPLES):
                # A ⊕ B = A + B + kappa * A * B
                m2_batch = m2_batch + free_energies[:, s] + (kappa * m2_batch * free_energies[:, s])
                
            total_m1 += m1_batch.sum().item()
            total_m2 += m2_batch.sum().item()
            total_images += current_batch_size

    return (total_m1 / total_images), (total_m2 / total_images)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Loading test dataset for additivity experiment...")
    test_loader = load_quick_testing_dataset()
    
    def get_kappa_float(dir_path):
        return float(os.path.basename(dir_path).split("_")[2])
        
    kappa_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "outputs_kappa_*")), key=get_kappa_float)
    
    with open(OUTPUT_FILE, "w") as f:
        # Strictly tab-separated header, no horizontal dashed lines
        f.write("Kappa\tStandard_Sum\tKappa_Pseudo_Sum\tDifference\n")
        
        for k_dir in kappa_dirs:
            kappa_str = os.path.basename(k_dir).split("_")[2]
            kappa_val = float(kappa_str)
            model_path = os.path.join(k_dir, f"cvae_kappa_{kappa_val}_dim_{LATENT_DIM}_latest.pth")
            
            if not os.path.exists(model_path):
                continue
                
            print(f"\nProcessing Kappa = {kappa_str}...")
            base_model = CoupledVariationalAutoencoder(3, LATENT_DIM, PRIOR_MEAN, PRIOR_VAR).to(DEVICE)
            state = torch.load(model_path, map_location=DEVICE, weights_only=True)
            base_model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state["model_state_dict"].items()})
            a_xz = compute_normalization_constant_term(kappa_val, LATENT_DIM).to(DEVICE)
            
            m1, m2 = evaluate_pseudo_additivity(base_model, test_loader, kappa_val, a_xz)
            diff = abs(m1 - m2)
            
            print(f"  Standard Sum:     {m1:.4f}")
            print(f"  Kappa Pseudo-Sum: {m2:.4f}")
            print(f"  Absolute Diff:    {diff:.4f}")
            
            # Tab-separated output
            f.write(f"{kappa_str}\t{m1:.6f}\t{m2:.6f}\t{diff:.6f}\n")
            f.flush()