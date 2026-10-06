import os
import glob
import zipfile
import argparse
import json
import numpy as np
import scipy.linalg
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms, models
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
parser = argparse.ArgumentParser(description="Calculate Robustness Metrics")
parser.add_argument("dataset", type=int, choices=[1, 2], help="1 = CelebA, 2 = MNIST")
args = parser.parse_args()

DATASET_NAME = "celeba_data" if args.dataset == 1 else "mnist_data"
BASE_DIR = f"/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/{DATASET_NAME}"
LOCAL_EVAL_DIR = '/content/evaluation_dataset'
DRIVE_ZIP_PATH = os.path.join(BASE_DIR, "evaluation_dataset.zip")

MODEL_DIRS = ['cvae', 'heavy_vae', 'beta_vae', 'prior_vae']
CORRUPTIONS = ["gaussian_noise", "motion_blur", "fog", "shot_noise"]
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM_FLOAT = transforms.Compose([transforms.ToTensor()])
TRANSFORM_UINT8 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
])

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_zip_if_present():
    originals_dir = os.path.join(LOCAL_EVAL_DIR, "originals")
    if not os.path.exists(originals_dir) or len(glob.glob(os.path.join(originals_dir, "*.png"))) == 0:
        if os.path.exists(DRIVE_ZIP_PATH):
            print(f"Extracting {DRIVE_ZIP_PATH} to local SSD ({LOCAL_EVAL_DIR})...")
            with zipfile.ZipFile(DRIVE_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall('/content')
            print("Extraction complete!")

def get_image_paths(folder_path):
    return sorted(glob.glob(os.path.join(folder_path, "*.png")))

def load_image_batch(paths, as_uint8=False):
    images = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        images.append(TRANSFORM_UINT8(img) if as_uint8 else TRANSFORM_FLOAT(img))
    return torch.stack(images).to(DEVICE)

def get_fully_processed_params(filepath):
    processed = set()
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            for line in file.readlines()[1:]:
                if line.strip():
                    processed.add(line.split()[0])
    return processed

def load_checkpoint(ckpt_path):
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            pass
    return {}

def save_checkpoint(state, ckpt_path):
    with open(ckpt_path, "w") as file:
        json.dump(state, file, indent=4)

# ==========================================
# METRIC EVALUATION FUNCTIONS
# ==========================================
def compute_paired_metrics(orig_paths, recon_paths):
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE)
    
    psnr_scores, ssim_scores, lpips_scores = [], [], []
    num_images = len(orig_paths)
    
    for i in tqdm(range(0, num_images, BATCH_SIZE), desc="    Paired Metrics", leave=False):
        batch_orig = orig_paths[i:i+BATCH_SIZE]
        batch_recon = recon_paths[i:i+BATCH_SIZE]

        orig_float = load_image_batch(batch_orig, as_uint8=False)
        recon_float = load_image_batch(batch_recon, as_uint8=False)

        psnr_scores.append(psnr_metric(recon_float, orig_float).item())
        ssim_scores.append(ssim_metric(recon_float, orig_float).item())
        lpips_scores.append(lpips_metric(recon_float, orig_float).item())

    return {
        "PSNR_mean": float(np.mean(psnr_scores)),
        "PSNR_std": float(np.std(psnr_scores)),
        "SSIM_mean": float(np.mean(ssim_scores)),
        "SSIM_std": float(np.std(ssim_scores)),
        "LPIPS_mean": float(np.mean(lpips_scores)),
        "LPIPS_std": float(np.std(lpips_scores))
    }

def compute_distribution_metrics(orig_paths, recon_paths):
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
    kid_metric = KernelInceptionDistance(subset_size=50, normalize=False).to(DEVICE)
    
    num_images = len(orig_paths)
    
    for i in tqdm(range(0, num_images, BATCH_SIZE), desc="    Dist Metrics", leave=False):
        batch_orig = orig_paths[i:i+BATCH_SIZE]
        batch_recon = recon_paths[i:i+BATCH_SIZE]

        orig_uint8 = load_image_batch(batch_orig, as_uint8=True)
        recon_uint8 = load_image_batch(batch_recon, as_uint8=True)

        fid_metric.update(orig_uint8, real=True)
        fid_metric.update(recon_uint8, real=False)
        
        kid_metric.update(orig_uint8, real=True)
        kid_metric.update(recon_uint8, real=False)

    fid_mean = fid_metric.compute().item()
    kid_mean, kid_std = kid_metric.compute()

    fid_metric.reset()
    kid_metric.reset()

    return {
        "FID_mean": float(fid_mean),
        "KID_mean": float(kid_mean.item()),
        "KID_std": float(kid_std.item())
    }

def extract_resnet_features(paths):
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE)
    resnet.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="    ResNet Feat", leave=False):
            batch_tensors = load_image_batch(paths[i:i+BATCH_SIZE], as_uint8=False)
            batch_tensors = normalize(batch_tensors)
            out = feature_extractor(batch_tensors)
            features.append(out.view(out.size(0), -1).cpu().numpy())
    return np.concatenate(features, axis=0)

def compute_frd_prec_rec(orig_paths, recon_paths):
    real_features = extract_resnet_features(orig_paths)
    fake_features = extract_resnet_features(recon_paths)
    
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu_real - mu_fake
    covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    frd_mean = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    
    nn_real = NearestNeighbors(n_neighbors=20, n_jobs=-1).fit(real_features)
    nn_fake = NearestNeighbors(n_neighbors=20, n_jobs=-1).fit(fake_features)
    
    real_distances, _ = nn_real.kneighbors(real_features)
    fake_distances, _ = nn_fake.kneighbors(fake_features)
    real_radii = real_distances[:, -1]
    fake_radii = fake_distances[:, -1]
    
    dist_fake_to_real = pairwise_distances(fake_features, real_features, n_jobs=-1)
    is_in_real_manifold = (dist_fake_to_real <= real_radii.reshape(1, -1)).any(axis=1)
    prec_mean = is_in_real_manifold.mean()
    
    is_in_fake_manifold = (dist_fake_to_real.T <= fake_radii.reshape(1, -1)).any(axis=1)
    rec_mean = is_in_fake_manifold.mean()
    
    return {
        "FRD_mean": float(frd_mean),
        "Prec_mean": float(prec_mean),
        "Rec_mean": float(rec_mean)
    }

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    extract_zip_if_present()
    print(f"Aggregating robustness metrics for {DATASET_NAME}...")
    
    orig_dir = os.path.join(LOCAL_EVAL_DIR, 'originals')
    orig_paths = get_image_paths(orig_dir)
    
    if not orig_paths:
        raise FileNotFoundError(f"No original images found in {orig_dir}!")

    for model_name in MODEL_DIRS:
        model_path = os.path.join(BASE_DIR, model_name)
        if not os.path.exists(model_path):
            continue
            
        print(f"\n==========================================")
        print(f"EVALUATING ROBUSTNESS FOR MODEL: {model_name.upper()}")
        print(f"==========================================")
        
        checkpoint_file = os.path.join(model_path, "robustness_metrics_checkpoint.json")
        checkpoint_state = load_checkpoint(checkpoint_file)
        
        param_dirs = glob.glob(os.path.join(model_path, "outputs_*"))
        param_dirs.sort(key=lambda p: float(os.path.basename(p).split("_")[2]))
        
        for corruption in CORRUPTIONS:
            print(f"\n  --- Processing Corruption: {corruption.upper()} ---")
            output_results_file = os.path.join(model_path, f"robustness_results_{corruption}.txt")
            fully_processed = get_fully_processed_params(output_results_file)
            
            if corruption not in checkpoint_state:
                checkpoint_state[corruption] = {}
                
            write_mode = "a" if os.path.exists(output_results_file) else "w"
            
            with open(output_results_file, write_mode) as results_file:
                if write_mode == "w":
                    header = f"{'Parameter':<10} {'FID':<8} {'KID_mean':<9} {'KID_std':<8} {'LPIPS_mean':<10} {'LPIPS_std':<10} {'SSIM_mean':<12} {'SSIM_std':<12} {'PSNR_mean':<10} {'PSNR_std':<9} {'FRD':<8} {'Prec':<8} {'Rec':<8} {'F0.5':<8}\n"
                    results_file.write(header)
                    
                for p_dir in param_dirs:
                    folder_name = os.path.basename(p_dir)
                    param_val = folder_name.split("_")[2] 
                    
                    if param_val in fully_processed:
                        print(f"  Skipping Parameter: {param_val} (Already processed).")
                        continue
                        
                    # Look for corrupted reconstructions on local SSD first, then Drive
                    recon_dir = os.path.join(LOCAL_EVAL_DIR, f"reconstructions_{corruption}_{folder_name.replace('outputs_', '')}")
                    if not os.path.exists(recon_dir):
                        recon_dir = os.path.join(p_dir, f"reconstructions_{corruption}")
                        
                    recon_paths = get_image_paths(recon_dir)
                    
                    if not recon_paths or len(recon_paths) != len(orig_paths):
                        print(f"  Skipping Parameter: {param_val} (Incomplete {corruption} reconstructions in {recon_dir}).")
                        continue
                        
                    print(f"\n    Evaluating Parameter = {param_val}...")
                    
                    if param_val not in checkpoint_state[corruption]:
                        checkpoint_state[corruption][param_val] = {}
                    p_data = checkpoint_state[corruption][param_val]
                    
                    if "PSNR_mean" not in p_data:
                        p_data.update(compute_paired_metrics(orig_paths, recon_paths))
                        save_checkpoint(checkpoint_state, checkpoint_file)
                    if "FID_mean" not in p_data:
                        p_data.update(compute_distribution_metrics(orig_paths, recon_paths))
                        save_checkpoint(checkpoint_state, checkpoint_file)
                    if "Rec_mean" not in p_data:
                        p_data.update(compute_frd_prec_rec(orig_paths, recon_paths))
                        save_checkpoint(checkpoint_state, checkpoint_file)
                    
                    prec = p_data['Prec_mean']
                    rec = p_data['Rec_mean']
                    f05_score = (1.25 * prec * rec) / ((0.25 * prec) + rec) if (prec + rec) > 0 else 0.0

                    row = f"{param_val:<10} {p_data['FID_mean']:<8.2f} {p_data['KID_mean']:<9.3f} {p_data['KID_std']:<8.3f} {p_data['LPIPS_mean']:<10.3f} {p_data['LPIPS_std']:<10.3f} {p_data['SSIM_mean']:<12.3f} {p_data['SSIM_std']:<12.3f} {p_data['PSNR_mean']:<10.2f} {p_data['PSNR_std']:<9.1f} {p_data['FRD_mean']:<8.2f} {p_data['Prec_mean']:<8.3f} {p_data['Rec_mean']:<8.3f} {f05_score:<8.3f}\n"
                    results_file.write(row)
                    results_file.flush()

    print(f"\nAll robustness evaluations complete across all models!")