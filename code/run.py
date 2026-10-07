import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, utils
from torchvision.datasets import CelebA
from PIL import Image
from tqdm import tqdm
import py7zr
import shutil
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# =============================================================================
# GLOBAL SETTINGS & REPRODUCIBILITY
# =============================================================================


# Force CPU usage and strict determinism
computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(False)


def set_strict_random_seeds(seed_value=42):
    """Ensures reproducible results across runs without crippling CPU performance."""
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

set_strict_random_seeds(42)


def capture_rng_state():
    """RNG state for the checkpoint, stored with tensors and plain ints so torch.load(weights_only=True) accepts it."""
    np_state = np.random.get_state()
    state = {
        "torch": torch.get_rng_state(),
        "numpy_key": torch.from_numpy(np_state[1].astype(np.int64)),
        "numpy_rest": [int(np_state[2]), int(np_state[3]), float(np_state[4])],
    }
    if torch.cuda.is_available():
        state["cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state):
    """Puts the RNG state saved in a checkpoint back, so a resumed run continues the same random stream.
    Older checkpoints have no state and keep the seed-42 start."""
    if not state:
        return
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(("MT19937", state["numpy_key"].cpu().numpy().astype(np.uint32), *state["numpy_rest"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


# =============================================================================
# CONFIGURATION INPUTS & MODE SELECTION
# =============================================================================

while True:
    try:
        dataset_input = input("Select dataset - CelebA (1) or MNIST (2): ")
        dataset_choice = int(dataset_input)
        if dataset_choice not in [1, 2]:
            raise ValueError("Dataset must be 1 or 2.")
            
        model_input = input("Select model - CVAE (1), Heavy-Tail VAE (2), Beta-VAE (3), Prior-VAE (4): ")
        model_choice = int(model_input)
        if model_choice not in [1, 2, 3, 4]:
            raise ValueError("Model must be 1, 2, 3, or 4.")
            
        if model_choice in [1, 2]:
            kappa_input = input("Please enter the value for kappa (e.g., 0.0, 1e-5, 0.1, 1.0): ")
            coupling_kappa = float(kappa_input)
            beta_val = 1.0
            param_str = f"kappa_{coupling_kappa}"
        else:
            beta_input = input("Please enter the value for beta (e.g., 1.0, 2.0, 5.0): ")
            beta_val = float(beta_input)
            coupling_kappa = 0.0
            param_str = f"beta_{beta_val}"
            
        dim_input = input("Please enter the latent dimension (e.g., 100): ")
        latent_dimension = int(dim_input)
        
        samples_input = input("Please enter the number of samples (e.g., 1, 5, 10): ")
        number_of_samples = int(samples_input)
        
        mode_input = input("Select mode - Train (1), Eval Set (2), Clean (3), Robustness (4), Stochastic Consistency (5), t-SNE (6), or Standard Free Energy (7): ")
        execution_mode = int(mode_input)
        if execution_mode not in [1, 2, 3, 4, 5, 6, 7]:
            raise ValueError("Mode must be 1, 2, 3, 4, 5, 6, or 7.")
            
        break 
    except ValueError as e:
        print(f"Invalid input: {e}. Please try again.")
        
# Dataset-Specific Routing
if dataset_choice == 1: # CelebA
    image_size = 128
    input_channels_count = 3
    base_data_directory = '/content/celeba_dataset'  # FAST LOCAL SSD
    base_output_folder = '/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/celeba_data'
elif dataset_choice == 2: # MNIST
    image_size = 32 
    input_channels_count = 1
    base_data_directory = '/content/mnist_dataset'  # FAST LOCAL SSD
    base_output_folder = '/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/mnist_data'

    
# Model-Specific Directory Routing
model_folders = {1: 'cvae', 2: 'heavy_vae', 3: 'beta_vae', 4: 'prior_vae'}
base_output_folder = os.path.join(base_output_folder, model_folders[model_choice])

# Hyperparameters based on Model Choice
beta_weight = beta_val if model_choice == 3 else 1.0
prior_variance_value = 1.0 / math.sqrt(beta_val) if model_choice == 4 else 1.0
prior_mean_value = 0.0

stretching_alpha = 2
number_of_epochs = 5
batch_size = 64
learning_rate = 5e-4
evaluation_set_size = 10000

# Directories
archive_path = os.path.join(base_data_directory, 'img_align_celeba.7z')
data_directory = os.path.join(base_data_directory, 'img_align_celeba/')
output_directory = os.path.abspath(os.path.join(base_output_folder, f'model_{param_str}_dim_{latent_dimension}_samples_{number_of_samples}'))

# Save image sets on fast local SSD during generation to avoid Drive FUSE bottlenecks
# Save ALL bulk evaluation image sets on fast local SSD (/content/)
local_eval_base = '/content/evaluation_dataset'
evaluation_originals_dir = os.path.join(local_eval_base, 'originals')
# Redirect reconstructions folder to local SSD
# The model name keeps CVAE and benchmark reconstructions apart on the shared SSD folder
evaluation_reconstructions_dir = os.path.join(local_eval_base, f'reconstructions_{model_folders[model_choice]}_{param_str}_dim_{latent_dimension}_samples_{number_of_samples}')


# Logs and analysis tables of every run sit in the model folder (one level above output_directory), named by the run's
# parameters, so a plotting program can read them all from one place. pipeline_state.py uses the same names.
model_data_directory = os.path.abspath(base_output_folder)
run_tag = f'{param_str}_dim_{latent_dimension}_samples_{number_of_samples}'


def results_file(result_type, file_name):
    """Path of a table inside <model folder>/results_<result_type>/ (the folder is created)."""
    folder = os.path.join(model_data_directory, f"results_{result_type}")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, file_name)

os.makedirs(output_directory, exist_ok=True)
if execution_mode == 2:
    
    # Restore the shared evaluation set (originals + corruptions) after a Colab reset.
    # The zip sits in the dataset folder (one level above the model folder), the same
    # place the shell scripts and the metric scripts use. It holds a top-level
    # evaluation_dataset/ folder, so it extracts into /content.
    eval_zip_path = os.path.join(os.path.dirname(base_output_folder), 'evaluation_dataset.zip')
    if os.path.exists(eval_zip_path) and not os.path.exists(evaluation_originals_dir):
        import zipfile
        print(f"Found {eval_zip_path}. Extracting to /content ...")
        with zipfile.ZipFile(eval_zip_path, 'r') as zip_ref:
            zip_ref.extractall('/content')
        print("Extraction complete!")
    
    os.makedirs(evaluation_originals_dir, exist_ok=True)
    os.makedirs(evaluation_reconstructions_dir, exist_ok=True)


# =============================================================================
# COUPLED MATHEMATICS FOUNDATIONS
# =============================================================================

def compute_coupled_logarithm(value_tensor, kappa):
    """Computes raw coupled logarithm without epsilon clamping."""
    if abs(kappa) < 1e-12:
        return torch.log(value_tensor)
    
    return (value_tensor.pow(kappa) - 1.0) / kappa


def compute_coupled_logarithm_from_log_space(log_value_tensor, kappa):
    """
    Computes the coupled logarithm directly from log space for numerical stability.
    """

    if abs(kappa) < 1e-12:
        return log_value_tensor

    kappa_scaled_log = kappa * log_value_tensor

    expm1_value = torch.expm1(kappa_scaled_log) #exp(x) -1 with taylor

    return expm1_value / kappa


def compute_coupled_sum(value_a, value_b, kappa):
    """
    Computes the generalized coupled sum.
    """
    return value_a + value_b + kappa * value_a * value_b


def compute_log_partition_function(log_determinant, kappa, alpha, dimension):
    """Computes the log of the partition function Z_kappa directly in float64 log-space."""
    precision_dtype = torch.float64
    if not isinstance(log_determinant, torch.Tensor):
        log_determinant = torch.tensor(log_determinant, dtype=precision_dtype)
    else:
        log_determinant = log_determinant.to(dtype=precision_dtype)
        
    kappa_tensor = torch.tensor(float(kappa), dtype=precision_dtype, device=log_determinant.device)
    dim_value = int(dimension)
    
    if alpha != 2:
        raise ValueError("Only alpha=2 is supported.")
    # Paper Eq. partitionFunctionCoupledGaussian, for the density (1 + kappa*Q/2)^-(1 + kappa*d/2)/kappa
    if float(kappa) <= -2.0 / dim_value:
        raise ValueError(f"kappa must be strictly greater than -2/d = {-2.0/dim_value}")

    two_pi_tensor = torch.tensor(2.0 * math.pi, dtype=precision_dtype, device=log_determinant.device)

    if torch.isclose(kappa_tensor, torch.tensor(0.0, dtype=precision_dtype), atol=1e-12) or torch.abs(kappa_tensor) < 1e-6:
        return 0.5 * log_determinant + 0.5 * dim_value * torch.log(two_pi_tensor)

    half_dim = dim_value / 2.0

    if float(kappa) > 0.0:
        # Gamma(1/k) / Gamma(1/k + d/2) * (2*pi/k)^(d/2)
        log_function_term = (torch.special.gammaln(1.0 / kappa_tensor)
                             - torch.special.gammaln(1.0 / kappa_tensor + half_dim)
                             + half_dim * torch.log(two_pi_tensor / kappa_tensor))
    else:
        # Gamma(1 - 1/k - d/2) / Gamma(1 - 1/k) * (-2*pi/k)^(d/2), compact support
        log_function_term = (torch.special.gammaln(1.0 - 1.0 / kappa_tensor - half_dim)
                             - torch.special.gammaln(1.0 - 1.0 / kappa_tensor)
                             + half_dim * torch.log(-two_pi_tensor / kappa_tensor))

    log_partition_total = 0.5 * log_determinant + log_function_term
    return log_partition_total



def sample_multivariate_coupled_gaussian(batch_size, latent_dim, kappa, dimension, device):
    """
    Samples from a multivariate coupled Gaussian distribution using a Gamma-based scale mixture,
    ensuring correct variance across any number of dimensions.
    """
    if kappa == 0:
        return torch.randn(batch_size, latent_dim, device=device)
    
    # Degrees of freedom
    nu = 2.0 / kappa
    
    # Draw scale mixture weights from a Gamma distribution
    gamma_dist = torch.distributions.Gamma(
        torch.tensor([nu / 2.0], device=device), 
        torch.tensor([nu / 2.0], device=device)
    )
    v = gamma_dist.sample((batch_size,)).squeeze(-1)

    # Draw standard d-dimensional normal samples (NO CLAMPING ON v)
    z = torch.randn(batch_size, latent_dim, device=device)

    # Apply raw scale mixture (will explode/divide by zero when kappa >= 1.0)
    return z / torch.sqrt(v).unsqueeze(1)


# =============================================================================
# DATASET AND DATALOADERS
# =============================================================================

class CelebAImageDataset(Dataset):
    """Loads and transforms CelebA image datasets from a local directory."""
    
    def __init__(self, directory_path, transform_pipeline=None):
        # Added sorted() to ensure deterministic file ordering across different machines!
        self.image_paths = sorted(glob.glob(os.path.join(directory_path, "*.jpg")))
        self.transform_pipeline = transform_pipeline
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_data = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform_pipeline:
            image_data = self.transform_pipeline(image_data)
        return image_data, 0
    
training_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

evaluation_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

if execution_mode in [1, 2, 5, 6, 7]:
    if dataset_choice == 1:
        # --- CELEBA DATASET ---
        expected_images = 202599
        drive_data_folder = '/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/celeba_dataset'
    
        # Check if images are already extracted locally
        local_jpgs = glob.glob(os.path.join(base_data_directory, "**", "*.jpg"), recursive=True)
    
        if len(local_jpgs) < expected_images:
            os.makedirs(base_data_directory, exist_ok=True)
            
            # Scan Google Drive folder for any zip or 7z archive
            drive_zips = glob.glob(os.path.join(drive_data_folder, "**", "*.zip"), recursive=True) + \
                        glob.glob(os.path.join(drive_data_folder, "**", "*.7z"), recursive=True)
            
            if drive_zips:
                archive_to_extract = drive_zips[0]
                print(f"Found drive archive ({archive_to_extract}). Extracting to fast local VM disk ({base_data_directory})...")
                
                if archive_to_extract.endswith('.zip'):
                    import zipfile
                    with zipfile.ZipFile(archive_to_extract, 'r') as zip_ref:
                        zip_ref.extractall(base_data_directory)
                elif archive_to_extract.endswith('.7z'):
                    with py7zr.SevenZipFile(archive_to_extract, mode='r') as archive:
                        archive.extractall(path=base_data_directory)
                print("Local extraction complete!")
            else:
                raise FileNotFoundError(
                    f"No zip/7z archive found in Google Drive ({drive_data_folder}). "
                    "Please upload img_align_celeba.zip to your Google Drive folder."
                )
                
            # Re-scan to find where the .jpg files landed
            local_jpgs = glob.glob(os.path.join(base_data_directory, "**", "*.jpg"), recursive=True)
            
        if not local_jpgs:
            raise RuntimeError(f"Extraction failed or no .jpg images found in {base_data_directory}")
            
        # Dynamically set data_directory to whichever folder contains the .jpg files
        data_directory = os.path.dirname(local_jpgs[0])
        print(f"Successfully located {len(local_jpgs)} CelebA images in: {data_directory}")
            
        full_training_dataset = CelebAImageDataset(directory_path=data_directory, transform_pipeline=training_transforms)
        full_evaluation_dataset = CelebAImageDataset(directory_path=data_directory, transform_pipeline=evaluation_transforms)
        
        total_dataset_length = len(full_training_dataset)
        training_split_size = int(0.7 * total_dataset_length)
        validation_split_size = int(0.15 * total_dataset_length)
        testing_split_size = total_dataset_length - training_split_size - validation_split_size
        
        dataset_indices = torch.randperm(total_dataset_length).tolist()
        
        training_subset = Subset(full_training_dataset, dataset_indices[:training_split_size])
        validation_subset = Subset(full_evaluation_dataset, dataset_indices[training_split_size : training_split_size + validation_split_size])
        testing_subset = Subset(full_evaluation_dataset, dataset_indices[training_split_size + validation_split_size :])
        
    elif dataset_choice == 2:
        # --- MNIST DATASET ---
        from torchvision.datasets import MNIST
        mnist_zip_drive = '/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/mnist_dataset/mnist_raw_data.zip'
    
        # Check if raw zip exists on Google Drive to extract locally
        if os.path.exists(mnist_zip_drive) and not os.path.exists(os.path.join(base_data_directory, 'MNIST')):
            print("Extracting MNIST zip from Google Drive to fast local VM disk...")
            import zipfile
            with zipfile.ZipFile(mnist_zip_drive, 'r') as zip_ref:
                zip_ref.extractall(base_data_directory)
            print("MNIST local extraction complete!")
        else:
            print("Downloading/Verifying MNIST dataset on fast local VM disk...")
            
        mnist_train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        mnist_eval_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
        # Download train and test sets directly to local SSD
        full_training_dataset = MNIST(root=base_data_directory, train=True, download=True, transform=mnist_train_transforms)
        full_evaluation_dataset = MNIST(root=base_data_directory, train=False, download=True, transform=mnist_eval_transforms)
        
        # MNIST is pre-split (60k train, 10k test). We split the 10k test set into validation and testing.
        training_subset = full_training_dataset
        total_eval_length = len(full_evaluation_dataset)
        validation_split_size = int(0.5 * total_eval_length)
        
        validation_subset = Subset(full_evaluation_dataset, list(range(validation_split_size)))
        testing_subset = Subset(full_evaluation_dataset, list(range(validation_split_size, total_eval_length)))
        
    # Create the dataloaders universally
    # JPEG decoding is the CPU cost. Leave one core for the main process and cap at 8,
    # since more workers stop helping once the GPU is the slower side.
    # sched_getaffinity respects the cores the VM actually gives us.
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
    cpu_workers = max(1, min(8, available_cpus - 1))
    use_cuda = torch.cuda.is_available()
    print(f"DataLoader: {cpu_workers} workers ({available_cpus} CPUs available), pin_memory={use_cuda}")
    training_dataloader = DataLoader(training_subset, batch_size=batch_size, shuffle=True, num_workers=cpu_workers,
                                     pin_memory=use_cuda, persistent_workers=True, prefetch_factor=4)
    validation_dataloader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False, num_workers=cpu_workers, pin_memory=use_cuda)
    testing_dataloader = DataLoader(testing_subset, batch_size=batch_size, shuffle=False, num_workers=cpu_workers, pin_memory=use_cuda)

# =============================================================================
# MODEL DEFINITION
# =============================================================================

class CoupledVariationalAutoencoder(nn.Module):
    """Convolutional Variational Autoencoder configured for CelebA."""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 100, prior_mean_val=0.0, prior_variance_val=1.0):
            super(CoupledVariationalAutoencoder, self).__init__()
        
            self.latent_dimension = latent_dim
            self.input_channels = input_channels
            self.register_buffer('prior_mean', torch.tensor(prior_mean_val))
            self.register_buffer('prior_variance', torch.tensor(prior_variance_val))
        
            spatial_dimension_after_encoder = image_size // 16
            flattened_feature_size = 256 * (spatial_dimension_after_encoder ** 2)
        
            self.encoder_network = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU()
            )
        
            self.fully_connected_mean = nn.Linear(flattened_feature_size, latent_dim)
            self.fully_connected_log_variance = nn.Linear(flattened_feature_size, latent_dim)
            self.decoder_projection = nn.Linear(latent_dim, flattened_feature_size)
        
            self.decoder_network = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )
        
            self.apply(self.initialize_network_weights)


    def encode(self, input_tensor: torch.Tensor):
            encoded_features = self.encoder_network(input_tensor)
            encoded_features = torch.flatten(encoded_features, start_dim=1)
            latent_mean = self.fully_connected_mean(encoded_features)
            latent_log_variance = self.fully_connected_log_variance(encoded_features)
            return latent_mean, latent_log_variance
    
    def decode(self, latent_samples):
        batch_size, samples, dimensions = latent_samples.shape
        flattened_samples = latent_samples.view(batch_size * samples, dimensions)
        
        spatial_dimension = image_size // 16
        projected_features = self.decoder_projection(flattened_samples)
        projected_features = projected_features.view(batch_size * samples, 256, spatial_dimension, spatial_dimension)
        
        reconstructed_images = self.decoder_network(projected_features)
        reconstructed_images = reconstructed_images.view(batch_size, samples, self.input_channels, image_size, image_size)
        return reconstructed_images

    def apply_reparameterization_trick(self, mean, log_variance, kappa, dimension, scale_covariance=True, num_samples=1):
        batch_size, dim = mean.shape
        standard_deviation = torch.exp(0.5 * log_variance) 
        effective_kappa = kappa
        
        if scale_covariance and kappa > 0:
            # Matches Appendix B derivation: \tilde{\Sigma} = \Sigma / (1 + \kappa)
            scaling_factor = torch.sqrt(torch.tensor(1.0 + kappa, dtype=standard_deviation.dtype, device=standard_deviation.device))
            standard_deviation = standard_deviation / scaling_factor
            effective_kappa = kappa / (1.0 + kappa)
            
        noise_samples = sample_multivariate_coupled_gaussian(batch_size * num_samples, dim, effective_kappa, dimension, mean.device)
        noise_samples = noise_samples.view(batch_size, num_samples, dim) 
        
        mean_expanded = mean.unsqueeze(1) 
        standard_deviation_expanded = standard_deviation.unsqueeze(1) 
        
        latent_z = mean_expanded + standard_deviation_expanded * noise_samples
        return latent_z

    def forward(self, input_tensor: torch.Tensor, current_kappa, dimension, num_samples=1, scale_covariance=True):
        latent_mean, latent_log_variance = self.encode(input_tensor)
        latent_samples = self.apply_reparameterization_trick(
            latent_mean, latent_log_variance, current_kappa, dimension, 
            scale_covariance=scale_covariance, num_samples=num_samples
        )
        reconstructed_output = self.decode(latent_samples)
        # Raw reconstructed output without bounding/clamping
        return reconstructed_output, latent_mean, latent_log_variance, latent_samples

    @staticmethod
    def initialize_network_weights(module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            init.ones_(module.weight)
            init.zeros_(module.bias)
            module.reset_running_stats()
            
# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
def compute_normalization_constant_term(kappa, dimension):
    """Computes the A_xz parameter required for reconstruction loss."""
    log_partition = compute_log_partition_function(0.0, kappa, alpha=2, dimension=dimension)
    
    exponent = 1.0 / (1.0 + float(kappa) * float(dimension) / 2.0)
    log_term = exponent * log_partition
    
    return compute_coupled_logarithm_from_log_space(log_term.to(torch.get_default_dtype()), kappa)


def compute_mahalanobis_reconstruction_loss(original_data, reconstructed_data, normalization_term, kappa):
    """Computes reconstruction loss utilizing the coupled Mahalanobis distance."""
    batch_size = original_data.size(0)
    flattened_original = original_data.view(batch_size, 1, -1)
    flattened_reconstructed = reconstructed_data.view(batch_size, -1, flattened_original.size(-1))
    
    difference = flattened_original - flattened_reconstructed 
    mahalanobis_distance = (difference ** 2).sum(dim=2) 
    
    # FIX: Multiply by 0.5 INSIDE the sum, and use kappa, not 2.0 * kappa
    loss = compute_coupled_sum(0.5 * mahalanobis_distance, normalization_term, kappa) 
    return loss.mean()


def compute_coupled_elbo_loss(reconstructed_data, original_data, posterior_mean, posterior_log_variance, 
                    prior_mean_val, prior_variance_val, kappa, dimension, normalization_term, beta_weight=1.0):
    """Computes the full Evidence Lower Bound utilizing Coupled Divergence."""
    batch_size, dim = posterior_mean.shape
    device = posterior_mean.device
    
    reconstruction_loss = compute_mahalanobis_reconstruction_loss(original_data, reconstructed_data, normalization_term, kappa)
    posterior_variance_matrix = torch.diag_embed(torch.exp(posterior_log_variance)) 
    prior_mean_tensor = torch.full_like(posterior_mean, prior_mean_val)       
    
    if kappa == 0.0:
        prior_variance_matrix_inverse = (1.0 / prior_variance_val) * torch.eye(dim, device=device).unsqueeze(0).expand(batch_size, dim, dim)
        mean_difference = (prior_mean_tensor - posterior_mean).unsqueeze(-1) 
        
        mahalanobis_distance = torch.bmm(torch.bmm(mean_difference.transpose(1, 2), prior_variance_matrix_inverse), mean_difference).squeeze() 
        trace_term = torch.einsum('bij,bji->b', prior_variance_matrix_inverse, posterior_variance_matrix) 
        log_determinant_posterior = torch.sum(posterior_log_variance, dim=1) 
        
        prior_variance_tensor = prior_variance_val.clone().detach().to(dtype=torch.get_default_dtype(), device=device)
        log_determinant_prior = dim * torch.log(prior_variance_tensor)
        
        kl_divergence = 0.5 * (trace_term + mahalanobis_distance - dim + log_determinant_prior - log_determinant_posterior)
        mean_kl_divergence = kl_divergence.mean()
        
    else:
        # FIX: information_kappa should just be kappa
        information_kappa = kappa
        prior_variance_matrix_inverse = (1.0 / prior_variance_val) * torch.eye(dim, device=device).unsqueeze(0).expand(batch_size, dim, dim)
        mean_difference = (prior_mean_tensor - posterior_mean).unsqueeze(-1)
        
        mahalanobis_distance = torch.bmm(torch.bmm(mean_difference.transpose(1, 2), prior_variance_matrix_inverse), mean_difference).squeeze()
        trace_term = torch.einsum('bij,bji->b', prior_variance_matrix_inverse, posterior_variance_matrix) 
        
        posterior_mean_64 = posterior_mean.double()
        posterior_log_variance_64 = posterior_log_variance.double()
        prior_variance_64 = prior_variance_val.clone().detach().to(dtype=torch.float64, device=device)
        
        log_determinant_posterior = torch.sum(posterior_log_variance_64, dim=1)
        log_determinant_prior = torch.full((batch_size,), dim, dtype=torch.float64, device=device) * torch.log(prior_variance_64)
        
        log_partition_posterior = compute_log_partition_function(log_determinant_posterior, kappa, alpha=2, dimension=dim) 
        log_partition_prior = compute_log_partition_function(log_determinant_prior, kappa, alpha=2, dimension=dim) 
        
        # FIX: Exponent matches 1 / (1 + kappa * d / 2)
        exponent = 1.0 / (1.0 + float(dimension) * float(kappa) / 2.0)
        log_term_posterior = exponent * log_partition_posterior
        log_term_prior = exponent * log_partition_prior
        
        # Direct raw evaluation without Taylor fallback masks
        term_a_posterior = compute_coupled_logarithm_from_log_space(log_term_posterior, information_kappa)
        term_a_prior = compute_coupled_logarithm_from_log_space(log_term_prior, information_kappa)
        
        
        component_1 = -0.5 * dimension * (1.0 + kappa * term_a_posterior.float())
        component_2 = 0.5 * (1.0 + kappa * term_a_prior.float()) * (mahalanobis_distance + trace_term)
        
        # FIX: Remove the 0.5 multipliers from A_q and A_p
        component_3 = -term_a_posterior.float() + term_a_prior.float()
        
        divergence = component_1 + component_2 + component_3
        mean_kl_divergence = divergence.mean()
        
    # BETA WEIGHT
    mean_kl_divergence = mean_kl_divergence * beta_weight
        
    average_posterior_mean = posterior_mean.mean()
    average_posterior_variance = torch.mean(torch.einsum('bii->b', posterior_variance_matrix))
    total_elbo = reconstruction_loss + mean_kl_divergence
    
    return total_elbo, reconstruction_loss, mean_kl_divergence, average_posterior_mean, average_posterior_variance, prior_mean_val, prior_variance_val

# =============================================================================
# TRAINING AND VALIDATION PIPELINES
# =============================================================================
def save_image_reconstructions(tensor_data, file_path, grid_rows=8):
    """Reshapes and saves image tensors reliably."""
    if tensor_data.dim() == 5:
        tensor_data = tensor_data.view(-1, input_channels_count, image_size, image_size)
    elif tensor_data.dim() == 4 and tensor_data.shape[1] != input_channels_count:
        raise ValueError(f"Expected shape (N, {input_channels_count}, H, W), but received {tensor_data.shape}")
    utils.save_image(tensor_data, file_path, nrow=grid_rows, normalize=False)


def execute_training_epoch(model, optimizer, dataloader, device, current_epoch, kappa, dimension, normalization_term, log_file, samples=1, global_batch_step=0, model_choice=1, beta_weight=1.0):
    model.train()
    accumulated_metrics = np.zeros(7)
    total_batches = len(dataloader)
    
    use_ie_scaling = (model_choice == 1)
    
    for batch_index, (image_batch, _) in enumerate(tqdm(dataloader, desc=f"Epoch {current_epoch} - Training")):
        image_batch = image_batch.to(device)
        optimizer.zero_grad()
        
        reconstructed_batch, latent_mean, latent_logvar, _ = model(
            image_batch, kappa, dimension, num_samples=samples, scale_covariance=use_ie_scaling
        )
        
        metrics = compute_coupled_elbo_loss(
            reconstructed_batch, image_batch, latent_mean, latent_logvar, 
            model.prior_mean, model.prior_variance, kappa, dimension, normalization_term, beta_weight
        )
        
        elbo_val = metrics[0]
        
        # EARLY STOPPING CHECK: Catch NaN or Inf instantly
        if torch.isnan(elbo_val) or torch.isinf(elbo_val):
            print(f"\n[CRASH DETECTED] NaN/Inf encountered at Epoch {current_epoch}, Batch {batch_index}! Terminating training run.")
            log_file.write(f"{current_epoch}\t{global_batch_step}\t{batch_index}\tnan\tnan\tnan\tnan\tnan\n")
            log_file.flush()
            return None, global_batch_step  # Return None to signal a numerical explosion
        
        elbo_v, recon_v, kl_v, mean_v, var_v, _, _ = metrics
        log_file.write(f"{current_epoch}\t{global_batch_step}\t{batch_index}\t{elbo_v:.6f}\t{recon_v:.6f}\t{kl_v:.6f}\t{mean_v:.6f}\t{var_v:.6f}\n")
        
        global_batch_step += 1
        
        metrics[0].backward()
        optimizer.step()
        
        accumulated_metrics += np.array([m.item() for m in metrics])
        
    return accumulated_metrics / total_batches, global_batch_step


def execute_validation_epoch(model, dataloader, device, current_epoch, kappa, dimension, normalization_term, samples=1):
    model.eval()
    accumulated_metrics_kappa = np.zeros(7)
    accumulated_metrics_kappa_q = np.zeros(7)
    
    with torch.no_grad():
        for batch_index, (image_batch, _) in enumerate(tqdm(dataloader, desc=f"Epoch {current_epoch} - Validation")):
            image_batch = image_batch.to(device)
            latent_mean, latent_logvar = model.encode(image_batch)
            
            # Sampling using standard kappa
            latent_samples_kappa = model.apply_reparameterization_trick(latent_mean, latent_logvar, kappa, dimension, scale_covariance=False, num_samples=samples)
            reconstructed_kappa = model.decode(latent_samples_kappa).clamp(min=1e-7, max=1.0 - 1e-7)
            metrics_kappa = compute_coupled_elbo_loss(
                reconstructed_kappa, image_batch, latent_mean, latent_logvar,
                model.prior_mean, model.prior_variance, kappa, dimension, normalization_term
            )
            accumulated_metrics_kappa += np.array([m.item() for m in metrics_kappa])
            
            # Sampling using scaled covariance (kappaQ)
            latent_samples_kappa_q = model.apply_reparameterization_trick(latent_mean, latent_logvar, kappa, dimension, scale_covariance=True, num_samples=samples)
            reconstructed_kappa_q = model.decode(latent_samples_kappa_q).clamp(min=1e-7, max=1.0 - 1e-7)
            metrics_kappa_q = compute_coupled_elbo_loss(
                reconstructed_kappa_q, image_batch, latent_mean, latent_logvar,
                model.prior_mean, model.prior_variance, kappa, dimension, normalization_term
            )
            accumulated_metrics_kappa_q += np.array([m.item() for m in metrics_kappa_q])
            
    return accumulated_metrics_kappa / len(dataloader), accumulated_metrics_kappa_q / len(dataloader)


def trim_log_to_epoch(log_path, last_saved_epoch):
    """Drops log rows from epochs after the last saved checkpoint.
    A crash between writing the log and saving the checkpoint (or in the middle of an epoch)
    leaves rows for an epoch that will be trained again. Removing them keeps one row per epoch."""
    if not os.path.exists(log_path):
        return
    with open(log_path, "r") as log_file:
        lines = log_file.readlines()
    kept = [lines[0]] if lines else []
    for line in lines[1:]:
        first_field = line.split("\t", 1)[0].strip()
        if first_field.isdigit() and int(first_field) <= last_saved_epoch:
            kept.append(line)
    if len(kept) != len(lines):
        print(f"Trimmed {len(lines) - len(kept)} log rows after epoch {last_saved_epoch} from {os.path.basename(log_path)}")
        with open(log_path, "w") as log_file:
            log_file.writelines(kept)

def generate_and_save_epoch_images(epoch, model, device, fixed_images_batch, kappa, dimension, output_dir):
    """Generates and saves reconstructions and random samples for both kappa and kappa_q."""
    model.eval()
    kappa_q = kappa / (1.0 + kappa)
    
    with torch.no_grad():
        # 1. Reconstructions using standard kappa
        latent_mean, latent_logvar = model.encode(fixed_images_batch)
        latent_z_kappa = model.apply_reparameterization_trick(latent_mean, latent_logvar, kappa, dimension, scale_covariance=False, num_samples=1)
        reconstruction_kappa = model.decode(latent_z_kappa).clamp(min=1e-7, max=1.0 - 1e-7)
        save_image_reconstructions(reconstruction_kappa.cpu(), os.path.join(output_dir, f"reconstruction_epoch_{epoch}_kappa.png"))
        
        # 2. Reconstructions using scaled covariance (kappa_q)
        latent_z_kappa_q = model.apply_reparameterization_trick(latent_mean, latent_logvar, kappa, dimension, scale_covariance=True, num_samples=1)
        reconstruction_kappa_q = model.decode(latent_z_kappa_q).clamp(min=1e-7, max=1.0 - 1e-7)
        save_image_reconstructions(reconstruction_kappa_q.cpu(), os.path.join(output_dir, f"reconstruction_epoch_{epoch}_kappa_q.png"))
        
        # 3. Random samples using standard kappa
        random_samples_kappa = sample_multivariate_coupled_gaussian(64, dimension, kappa, dimension, device)
        random_samples_kappa = random_samples_kappa.unsqueeze(1) 
        generated_images_kappa = model.decode(random_samples_kappa).clamp(min=1e-7, max=1.0 - 1e-7)
        save_image_reconstructions(generated_images_kappa.cpu(), os.path.join(output_dir, f"samples_epoch_{epoch}_kappa.png"))
        
        # 4. Random samples using scaled covariance (kappa_q)
        random_samples_kappa_q = sample_multivariate_coupled_gaussian(64, dimension, kappa_q, dimension, device)
        random_samples_kappa_q = random_samples_kappa_q.unsqueeze(1)
        generated_images_kappa_q = model.decode(random_samples_kappa_q).clamp(min=1e-7, max=1.0 - 1e-7)
        save_image_reconstructions(generated_images_kappa_q.cpu(), os.path.join(output_dir, f"samples_epoch_{epoch}_kappa_q.png"))


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def generate_strict_evaluation_dataset(model, dataset, device, kappa, dimension, num_images):
    """Deterministically saves original and reconstructed images for 1-to-1 metric comparison."""
    model.eval()
    
    actual_num_images = min(num_images, len(dataset))
    
    # Check if already generated
    if len(glob.glob(os.path.join(evaluation_reconstructions_dir, "*.png"))) >= actual_num_images:
        print(f"Evaluation reconstructions already exist in {evaluation_reconstructions_dir}. Skipping generation.")
        return
    
    eval_loader = DataLoader(Subset(dataset, range(actual_num_images)), batch_size=16, shuffle=False)
    
    global_idx = 0
    with torch.no_grad():
        for image_batch, _ in tqdm(eval_loader, desc="Generating Evaluation Images"):
            image_batch = image_batch.to(device)
            latent_mean, latent_logvar = model.encode(image_batch)
            
            # FORCE num_samples=1 FOR 1-to-1 METRIC COMPARISON
            latent_z = model.apply_reparameterization_trick(latent_mean, latent_logvar, kappa, dimension, scale_covariance=False, num_samples=1)
            reconstructed_batch = model.decode(latent_z).clamp(min=1e-7, max=1.0 - 1e-7)
            
            for i in range(image_batch.size(0)):
                orig_path = os.path.join(evaluation_originals_dir, f"img_{global_idx:04d}.png")
                recon_path = os.path.join(evaluation_reconstructions_dir, f"img_{global_idx:04d}.png")
                
                if not os.path.exists(orig_path):
                    utils.save_image(image_batch[i].cpu(), orig_path, normalize=False)
                    
                # reconstructed_batch has shape (Batch, Samples, C, H, W). We explicitly select sample 0.
                utils.save_image(reconstructed_batch[i, 0].cpu(), recon_path, normalize=False)
                global_idx += 1
                
    print(f"Successfully saved {global_idx} images to {evaluation_reconstructions_dir}")
    
def generate_corrupted_reconstructions(model, device, kappa, dimension):
    model.eval()
    corruptions = ["gaussian_noise", "motion_blur", "fog", "shot_noise"]
    # Point directly to fast local SSD
    base_eval_dir = '/content/evaluation_dataset'
    
    with torch.no_grad():
        for corruption in corruptions:
            corrupted_input_dir = os.path.join(base_eval_dir, corruption)
            target_recon_dir = os.path.join(base_eval_dir, f"reconstructions_{corruption}_{model_folders[model_choice]}_{param_str}_dim_{dimension}_samples_{number_of_samples}")
            
            
            if not os.path.exists(corrupted_input_dir):
                print(f"Skipping {corruption}: Folder not found.")
                continue
            
            image_paths = sorted(glob.glob(os.path.join(corrupted_input_dir, "*.png")))
            
            # Check if already generated
            if os.path.exists(target_recon_dir) and len(glob.glob(os.path.join(target_recon_dir, "*.png"))) >= len(image_paths):
                print(f"Skipping {corruption}: Reconstructions already generated.")
                continue
            
            os.makedirs(target_recon_dir, exist_ok=True)
            print(f"Generating {len(image_paths)} reconstructions for {corruption}...")
            
            for i in tqdm(range(0, len(image_paths), 16), desc=f"Processing {corruption}"):
                batch_paths = image_paths[i:i+16]
                
                images = []
                for path in batch_paths:
                    img = Image.open(path).convert("L" if dataset_choice == 2 else "RGB")
                    img = evaluation_transforms(img)
                    images.append(img)
                    
                image_batch = torch.stack(images).to(device)
                latent_mean, latent_logvar = model.encode(image_batch)
                
                # FORCE num_samples=1
                latent_z = model.apply_reparameterization_trick(latent_mean, latent_logvar, kappa, dimension, scale_covariance=False, num_samples=1)
                reconstructed_batch = model.decode(latent_z).clamp(min=1e-7, max=1.0 - 1e-7)
                
                for j, path in enumerate(batch_paths):
                    filename = os.path.basename(path)
                    recon_path = os.path.join(target_recon_dir, filename)
                    # Explicitly select sample 0
                    utils.save_image(reconstructed_batch[j, 0].cpu(), recon_path, normalize=False)
                    
def execute_stochastic_consistency_analysis(model, dataset, device, kappa, dimension):
    """
    Feeds a single image into the model 10,000 times in batches to analyze stochastic consistency.
    Saves an 8x8 visual grid (16 best, 32 median, 16 worst), a list of all Mahalanobis radii, 
    and a normalized histogram estimation without empty bins.
    """
    model.eval()
    print(f"Executing 10,000 stochastic consistency samples for kappa {kappa}...")
    
    # Grab the very first image from the testing dataset
    single_image, _ = dataset[158]
    
    total_samples = 100000
    batch_size = 100  # Process in batches of 100 to prevent memory overflow
    iterations = total_samples // batch_size
    
    # Replicate the single image to match the batch size
    image_batch = single_image.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
    flattened_original = image_batch.view(batch_size, -1)
    
    all_mahalanobis_radii = []
    all_latent_zs = []
    
    with torch.no_grad():
        for _ in tqdm(range(iterations), desc="Generating Stochastic Samples"):
            # Encode the image (identical for all copies)
            latent_mean, latent_logvar = model.encode(image_batch)
            
            # Apply stochastic sampling
            latent_z = model.apply_reparameterization_trick(latent_mean, latent_logvar, kappa, dimension, scale_covariance=False, num_samples=1)
            
            # Store the tiny latent vectors on CPU to save RAM
            all_latent_zs.append(latent_z.cpu())
            
            # Decode the unique sampled vectors to compute distance
            reconstructed_batch = model.decode(latent_z).clamp(min=1e-7, max=1.0 - 1e-7)
            
            # Calculate the pure Mahalanobis distance
            flattened_reconstructed = reconstructed_batch.view(batch_size, -1)
            difference = flattened_original - flattened_reconstructed 
            batch_radii = (difference ** 2).sum(dim=1)
            
            # Store the radii
            all_mahalanobis_radii.extend(batch_radii.cpu().tolist())
            
    # Combine all 10,000 latent vectors into a single tensor
    full_latent_z = torch.cat(all_latent_zs, dim=0)
    
    # ---------------------------------------------------------
    # GENERATE THE REPRESENTATIVE 8x8 GRID
    # ---------------------------------------------------------
    # Get the sorted indices based on the Mahalanobis radii
    sorted_indices = np.argsort(all_mahalanobis_radii)
    
    # 1. 16 smallest radii
    smallest_idx = sorted_indices[:16]
    # 2. 32 median radii
    mid_point = total_samples // 2
    middle_idx = sorted_indices[mid_point - 16 : mid_point + 16]
    # 3. 16 largest radii
    largest_idx = sorted_indices[-16:]
    
    # Concatenate them sequentially
    selected_indices = np.concatenate([smallest_idx, middle_idx, largest_idx])
    
    with torch.no_grad():
        # Extract the 64 chosen latent vectors and decode them
        selected_z = full_latent_z[selected_indices].to(device)
        grid_images = model.decode(selected_z).clamp(min=1e-7, max=1.0 - 1e-7)
        
        # Save the structured 8x8 grid
        grid_path = os.path.join(output_directory, f"stochastic_reconstructions_kappa_{kappa}.png")
        save_image_reconstructions(grid_images.cpu(), grid_path, grid_rows=8)
        print(f"\nSaved structured 8x8 stochastic grid (16 min, 32 median, 16 max radii) to {grid_path}")
        
    # ---------------------------------------------------------
    # SAVE TEXT DATA (RADII & HISTOGRAM)
    # ---------------------------------------------------------
    # 1. Save the raw radii to a text file
    txt_path = results_file("stochastic_radii", f"stochastic_radii_{run_tag}.txt")
    with open(txt_path, "w") as f:
        f.write("Sample_Index\tMahalanobis_Radius\n")
        for i, radius in enumerate(all_mahalanobis_radii):
            f.write(f"{i}\t{radius:.4f}\n")
    print(f"Saved {total_samples} Mahalanobis radii to {txt_path}")
    
    # 2. Calculate and save the normalized histogram
    hist_counts, bin_edges = np.histogram(all_mahalanobis_radii, bins=100)
    normalized_freqs = hist_counts / total_samples
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    hist_path = results_file("stochastic_radii_histogram", f"stochastic_radii_histogram_{run_tag}.txt")
    with open(hist_path, "w") as f:
        f.write("Bin_Center\tNormalized_Frequency\n")
        for i in range(len(hist_counts)):
            # Only write lines where the frequency is greater than zero
            if normalized_freqs[i] > 0:
                f.write(f"{bin_centers[i]:.4f}\t{normalized_freqs[i]:.6f}\n")
    print(f"Saved normalized, filtered histogram estimation to {hist_path}")
    
def generate_tsne_visualization(model, dataloader, device, kappa, output_dir, max_samples=5000):
    """Extracts latent representations and generates a 2D t-SNE scatter plot."""
    model.eval()
    print(f"Generating t-SNE visualization for kappa {kappa}...")
    
    latent_vectors = []
    labels_list = []
    
    with torch.no_grad():
        for image_batch, label_batch in tqdm(dataloader, desc="Extracting Latent Features"):
            image_batch = image_batch.to(device)
            # Use the encoded mean (the cluster center) for visualization
            latent_mean, _ = model.encode(image_batch)
            
            latent_vectors.append(latent_mean.cpu().numpy())
            labels_list.append(label_batch.cpu().numpy())
            
            if sum(len(v) for v in latent_vectors) >= max_samples:
                break
            
    # Flatten the lists and truncate to max_samples
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:max_samples]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]
    
    # Check if we need t-SNE or if the data is already 2D
    if latent_vectors.shape[1] == 2:
        print("Latent dimension is exactly 2. Skipping t-SNE and plotting raw latent space...")
        latent_2d = latent_vectors
        plot_title = f"Raw 2D Latent Space (Kappa = {kappa})"
        x_label = "Latent Dimension 1"
        y_label = "Latent Dimension 2"
    else:
        print("Running t-SNE dimensionality reduction (this will take a moment)...")
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vectors)
        plot_title = f"t-SNE Latent Space Visualization (Kappa = {kappa})"
        x_label = "t-SNE 1st dimension"
        y_label = "t-SNE 2nd dimension"
        
    # Configure Matplotlib for Times New Roman with Linux fallbacks, medium size (12pt)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['font.size'] = 12

    plt.figure(figsize=(8, 6))
    
    # For MNIST, use the 'tab10' colormap which has exactly 10 distinct colors
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=15)
    
    # Add a colorbar with discrete ticks for digits 0-9
    colorbar = plt.colorbar(scatter, ticks=range(10))
    colorbar.set_label('Digit Class')
    
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Save the figure
    plot_path = os.path.join(output_dir, f"tsne_plot_kappa_{kappa}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE plot to {plot_path}")
    
def execute_standard_free_energy_analysis(model, dataloader, device, kappa, dimension, output_dir, num_mc_samples=10):
    """
    Standard (kappa = 0) negative ELBO, scored the same way for every model so the rows can be compared
    (paper Lemma common_gaussian_evaluation). Only the encoder means and scales and the decoder mean
    come from the trained model:
      q(z|x) = N(mu_q, Sigma_q), the encoder's scale used as a covariance. For the CVAE this is the
               covariance of the independent-equals distribution that training sampled from.
      p(z)   = N(mu_p, sigma_p^2 I), the model's prior location and scale (sigma_p = 1 except Prior-VAE).
      p(x|z) = N(x_bar(z), sigma_x^2 I), with one sigma_x per model fitted by maximum likelihood,
               sigma_x^2 = mean squared pixel error over the evaluation set and the MC samples.
    The KL has a closed form. The reconstruction term is a Monte Carlo average over z ~ q.
    """
    model.eval()
    print(f"\n[Mode 7] Computing the common Gaussian free energy for kappa={kappa} using {num_mc_samples} MC samples per image...")

    data_space_dim = input_channels_count * image_size * image_size
    prior_mean = model.prior_mean.double()
    prior_variance = model.prior_variance.double()

    total_kl_div = 0.0
    total_squared_error = 0.0
    total_samples_processed = 0

    with torch.no_grad():
        for image_batch, _ in tqdm(dataloader, desc="Standard Free Energy Evaluation"):
            image_batch = image_batch.to(device)
            batch_size = image_batch.size(0)

            latent_mean, latent_logvar = model.encode(image_batch)

            # Closed-form KL(N(mu_q, diag var_q) || N(mu_p, sigma_p^2 I)), in float64
            mean_64 = latent_mean.double()
            logvar_64 = latent_logvar.double()
            batch_kl = 0.5 * torch.sum(
                torch.exp(logvar_64) / prior_variance
                + (mean_64 - prior_mean) ** 2 / prior_variance
                - 1.0 + torch.log(prior_variance) - logvar_64,
                dim=1)

            # Gaussian samples z ~ N(mu_q, diag var_q), whatever kappa the model was trained with
            latent_samples = model.apply_reparameterization_trick(
                latent_mean, latent_logvar, 0.0, dimension, scale_covariance=False, num_samples=num_mc_samples
            )  # Shape: (Batch, Samples, Latent_Dim)
            reconstructed_batch = model.decode(latent_samples)  # Shape: (Batch, Samples, C, H, W)
            orig_expanded = image_batch.unsqueeze(1).expand(-1, num_mc_samples, -1, -1, -1)
            squared_error = torch.sum((orig_expanded - reconstructed_batch).double() ** 2, dim=(2, 3, 4))  # (Batch, Samples)

            total_kl_div += torch.sum(batch_kl).item()
            total_squared_error += torch.sum(torch.mean(squared_error, dim=1)).item()
            total_samples_processed += batch_size

    # Maximum-likelihood decoder variance. With it, the average Gaussian reconstruction loss
    # (d_d/2) log(2 pi sigma_x^2) + Q / (2 sigma_x^2) becomes (d_d/2) (log(2 pi sigma_x^2) + 1).
    decoder_variance = total_squared_error / (total_samples_processed * data_space_dim)
    decoder_sigma = math.sqrt(decoder_variance)
    avg_recon = 0.5 * data_space_dim * (math.log(2.0 * math.pi * decoder_variance) + 1.0)
    avg_kl = total_kl_div / total_samples_processed
    avg_fe = avg_recon + avg_kl

    print(f"Standard Free Energy Analysis Complete:")
    print(f"  * Decoder sigma (fitted): {decoder_sigma:.6f}")
    print(f"  * Avg Standard Free Energy: {avg_fe:.6f}")
    print(f"  * Avg Standard Recon Loss: {avg_recon:.6f}")
    print(f"  * Avg Standard KL Divergence: {avg_kl:.6f}")

    # One table per model folder with a row per run, sorted from the smallest to the largest parameter.
    # A rerun of the same run replaces its row.
    txt_path = results_file("standard_free_energy", "standard_free_energy.txt")
    param_name = "Kappa" if model_choice in (1, 2) else "Beta"
    header = f"{param_name}\tDim\tSamples\tStandard_Free_Energy\tStandard_Recon_Loss\tStandard_KL_Divergence\tDecoder_Sigma"
    rows = {}
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f.read().splitlines()[1:]:
                fields = line.split("\t")
                if len(fields) == 7:
                    rows[(float(fields[0]), int(fields[1]), int(fields[2]))] = fields
    rows[(float(kappa), int(dimension), int(number_of_samples))] = [
        str(kappa), str(dimension), str(number_of_samples), f"{avg_fe:.6f}", f"{avg_recon:.6f}", f"{avg_kl:.6f}", f"{decoder_sigma:.6f}"]
    tmp_path = txt_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(header + "\n")
        for key in sorted(rows, key=lambda k: (k[1], k[2], k[0])):
            f.write("\t".join(rows[key]) + "\n")
    os.replace(tmp_path, txt_path)
    print(f"Saved results to {txt_path}")
    

if __name__ == "__main__":
    if execution_mode == 3:
        print(f"Cleaning up extracted dataset to save disk space...")
        try:
            if os.path.exists(data_directory):
                import shutil
                shutil.rmtree(data_directory)
                print("Cleanup successful. Uncompressed folder deleted.")
        except Exception as e:
            print(f"Warning: Could not delete the dataset directory. Error: {e}")
        exit()
        
    
    # --- INITIALIZE MODEL FOR MODES 1 & 2 ---
    print(f"Initializing Model on {computation_device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")
    base_model = CoupledVariationalAutoencoder(
            input_channels=input_channels_count, 
            latent_dim=latent_dimension, 
            prior_mean_val=prior_mean_value, 
            prior_variance_val=prior_variance_value
        ).to(computation_device)
    
    network_optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
    starting_epoch = 1
    
    # Dynamically construct the specific filename based on the chosen model
    model_prefix = model_folders[model_choice]  # e.g., 'cvae', 'heavy_vae', 'beta_vae'
    model_filename = f"{model_prefix}_{param_str}_dim_{latent_dimension}_latest.pth"
    model_checkpoint_path = os.path.join(output_directory, model_filename)
    
    if os.path.exists(model_checkpoint_path):
        saved_state = torch.load(model_checkpoint_path, map_location=computation_device, weights_only=True)
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in saved_state["model_state_dict"].items()}
        base_model.load_state_dict(clean_state_dict)
        network_optimizer.load_state_dict(saved_state["optimizer_state_dict"])
        starting_epoch = saved_state["epoch"] + 1
        restore_rng_state(saved_state.get("rng_state"))
        print(f"Resuming from epoch {starting_epoch} using weights from {model_filename}")
        
    if execution_mode == 1:
        ref_images_list = []
        for imgs, _ in validation_dataloader:
            ref_images_list.append(imgs)
            if sum(b.size(0) for b in ref_images_list) >= 64:
                break
        reference_images_batch = torch.cat(ref_images_list, dim=0)[:64].to(computation_device)
        
        data_space_dimension = input_channels_count * image_size * image_size
        normalization_constant_a_xz = compute_normalization_constant_term(coupling_kappa, data_space_dimension).to(computation_device)
        
        
        epoch_log_path = results_file("epoch_log", f"epoch_log_{run_tag}.txt")
        batch_log_path = results_file("batch_log", f"batch_log_{run_tag}.txt")

        # Training restarts at starting_epoch, so rows of later epochs are leftovers of an interrupted run.
        trim_log_to_epoch(epoch_log_path, starting_epoch - 1)
        trim_log_to_epoch(batch_log_path, starting_epoch - 1)

        with open(epoch_log_path, "a") as epoch_logger, open(batch_log_path, "a") as batch_logger:
            if os.path.getsize(epoch_log_path) == 0:
                epoch_logger.write("Epoch\tTrainELBO_kappaQ\tValELBO_kappa\tValELBO_kappaQ\tTrainRecon_kappaQ\tValRecon_kappa\tValRecon_kappaQ\tTrainKLD_kappaQ\tValKLD_kappa\tValKLD_kappaQ\tTrainPostMean\tValPostMean_kappa\tValPostMean_kappaQ\tTrainPostVar\tValPostVar_kappa\tValPostVar_kappaQ\tPriorMean\tPriorVar\n")
            if os.path.getsize(batch_log_path) == 0:
                batch_logger.write("Epoch\tCumulatedBatch\tBatch\tELBO\tRecon\tKLD\tPostMean\tPostVar\n")
                
            total_batches_per_epoch = len(training_dataloader)
            global_batch_step = (starting_epoch - 1) * total_batches_per_epoch
            
            for current_epoch in range(starting_epoch, number_of_epochs + 1):
                training_results, global_batch_step = execute_training_epoch(
                    base_model, network_optimizer, training_dataloader, computation_device, 
                    current_epoch, coupling_kappa, latent_dimension, normalization_constant_a_xz, batch_logger, 
                    samples=number_of_samples, global_batch_step=global_batch_step, 
                    model_choice=model_choice, beta_weight=beta_weight
                )
    
                # IF NAN/INF OCCURRED: fill this and every remaining epoch with 'nan' rows and stop training,
                # so the log has one row per epoch without spending GPU time on a diverged run
                if training_results is None:
                    for nan_epoch in range(current_epoch, number_of_epochs + 1):
                        epoch_logger.write(f"{nan_epoch}" + "\tnan" * 17 + "\n")
                    epoch_logger.flush()
                    print(f"Skipping remaining epochs for kappa={coupling_kappa} due to numerical crash.")
                    exit(0)  # Exit clean with code 0 so the bash script moves to the next parameter
                    
                validation_results_kappa, validation_results_kappa_q = execute_validation_epoch(
                    base_model, validation_dataloader, computation_device, 
                    current_epoch, coupling_kappa, latent_dimension, normalization_constant_a_xz, samples=number_of_samples
                )
    
                epoch_logger.write(f"{current_epoch}\t{training_results[0]:.6f}\t{validation_results_kappa[0]:.6f}\t{validation_results_kappa_q[0]:.6f}\t{training_results[1]:.6f}\t{validation_results_kappa[1]:.6f}\t{validation_results_kappa_q[1]:.6f}\t{training_results[2]:.6f}\t{validation_results_kappa[2]:.6f}\t{validation_results_kappa_q[2]:.6f}\t{training_results[3]:.6f}\t{validation_results_kappa[3]:.6f}\t{validation_results_kappa_q[3]:.6f}\t{training_results[4]:.6f}\t{validation_results_kappa[4]:.6f}\t{validation_results_kappa_q[4]:.6f}\t{training_results[5]:.6f}\t{training_results[6]:.6f}\n")
                epoch_logger.flush()
    
                generate_and_save_epoch_images(current_epoch, base_model, computation_device, reference_images_batch, coupling_kappa, latent_dimension, output_directory)
    
                batch_logger.flush()  # batch rows are written once per epoch, before the checkpoint
                torch.save({
                    "epoch": current_epoch, 
                    "model_state_dict": base_model.state_dict(), 
                    "optimizer_state_dict": network_optimizer.state_dict(),
                    "rng_state": capture_rng_state()
                }, model_checkpoint_path)
                
    elif execution_mode == 2:
        generate_strict_evaluation_dataset(base_model, testing_subset, computation_device, coupling_kappa, latent_dimension, evaluation_set_size)
        
    elif execution_mode == 4:
        generate_corrupted_reconstructions(base_model, computation_device, coupling_kappa, latent_dimension)
        
    elif execution_mode == 5:
        execute_stochastic_consistency_analysis(base_model, testing_subset, computation_device, coupling_kappa, latent_dimension)
        
    elif execution_mode == 6:
        generate_tsne_visualization(base_model, testing_dataloader, computation_device, coupling_kappa, output_directory)
        
    elif execution_mode == 7:
        execute_standard_free_energy_analysis(base_model, testing_dataloader, computation_device, coupling_kappa, latent_dimension, output_directory)








        



        
