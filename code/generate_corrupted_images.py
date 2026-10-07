import os
import sys
import glob
import cv2
import zipfile
import argparse
import math as m
import numpy as np
import noise
from tqdm import tqdm

# Parse dataset argument
parser = argparse.ArgumentParser(description="Generate Corrupted Images")
parser.add_argument("dataset", type=int, choices=[1, 2], help="1 = CelebA, 2 = MNIST")
args = parser.parse_args()

DATASET_NAME = "celeba_data" if args.dataset == 1 else "mnist_data"
BASE_DRIVE_DIR = f"/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/{DATASET_NAME}"
DRIVE_ZIP_PATH = os.path.join(BASE_DRIVE_DIR, "evaluation_dataset.zip")

LOCAL_EVAL_DIR = '/content/evaluation_dataset'
ORIGINALS_DIR = os.path.join(LOCAL_EVAL_DIR, 'originals')

# ==========================================
# NOISE GENERATION FUNCTIONS
# ==========================================
def gaussian_noise(img, mean=0, std=25):
    float_img = img.astype(np.float32)
    gauss_noise = np.random.normal(mean, std, float_img.shape)
    noisy_img = float_img + gauss_noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def Gauss1D(x, sigma=1):
    return np.exp(-(np.square(x))/(2*(sigma**2)))/(sigma*m.sqrt(2*m.pi))

def steerblur(kdim, majors=1, minors=0.5, theta=0):
    khw = (kdim-1)/2
    xy = np.zeros((2,kdim))
    xy[0,:] = np.arange(-khw,khw+1,1)
    xy[1,:] = np.arange(-khw,khw+1,1)
    R = np.asarray(((np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))),
                    (np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)))))
    gausskern = np.zeros((kdim,kdim))
    for i in range(kdim):
        for j in range(kdim):
            xyp = np.matmul(R,np.asarray((xy[0,i],xy[1,j])))
            gausskern[j,i] = Gauss1D(xyp[0], majors)*Gauss1D(xyp[1], minors)
    return gausskern

def motionnoise(img, kdim=11, majors=3, minors=1.5, theta=0):
    gausskern = steerblur(kdim, majors, minors, theta)
    return cv2.filter2D(img, -1, gausskern)

def fognoise(img, octaves=8, freq_base=16.0, alpha=0.7, base=0.95, randomize=True):
    h, w = img.shape[:2]
    fog = np.zeros((h,w))
    freq = freq_base * octaves
    off_x, off_y = np.random.randint(255, size=2) if randomize else (0, 0)
    for y in range(h):
        for x in range(w):
            fog[y,x] = noise.snoise2((y + off_y) / freq, (x + off_x) / freq, octaves) * 127.0 + 128.0
    fog = (fog - fog.min()) / (fog.max() - fog.min() + 1e-8)
    
    if img.ndim == 2:
        img_3d = img[:, :, None]
    else:
        img_3d = img
        
    if base > alpha:
        fogimg = (base-alpha)*img_3d + alpha*(img_3d*(1-fog[:,:,None])+250*fog[:,:,None])
    else:
        fogimg = alpha*(img_3d*(1-fog[:,:,None])+250*fog[:,:,None])
    return fogimg.squeeze().astype('int')

def shotnoise(img, peak=0.5, base=0.8):
    noise_mask = np.random.poisson(img / 255.0 * peak) / peak * 255
    noise_mask = noise_mask / (noise_mask.max() + 1e-8)
    shotimg = base*((1-noise_mask)*img) + noise_mask
    return shotimg.astype('int')

CORRUPTIONS = {
    "gaussian_noise": lambda img: gaussian_noise(img, std=25),
    "motion_blur": lambda img: np.clip(motionnoise(img, kdim=15, majors=5, minors=1.5, theta=45), 0, 255).astype(np.uint8),
    "fog": lambda img: np.clip(fognoise(img, octaves=4, freq_base=8, alpha=0.8), 0, 255).astype(np.uint8),
    "shot_noise": lambda img: np.clip(shotnoise(img, peak=0.5, base=0.8), 0, 255).astype(np.uint8)
}

if __name__ == "__main__":
    if not os.path.exists(ORIGINALS_DIR) or len(glob.glob(os.path.join(ORIGINALS_DIR, "*.png"))) == 0:
        if os.path.exists(DRIVE_ZIP_PATH):
            print(f"Extracting evaluation dataset from Drive ({DRIVE_ZIP_PATH}) to local SSD...")
            with zipfile.ZipFile(DRIVE_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall('/content')
            print("Extraction complete!")
        else:
            raise FileNotFoundError(f"Originals not found at {ORIGINALS_DIR} and no zip backup at {DRIVE_ZIP_PATH}.")
            
    image_paths = sorted(glob.glob(os.path.join(ORIGINALS_DIR, "*.png")))
    print(f"\n==========================================")
    print(f"GENERATING CORRUPTIONS FOR DATASET: {DATASET_NAME.upper()}")
    print(f"Found {len(image_paths)} original images.")
    print(f"==========================================")

    for corruption_name in CORRUPTIONS.keys():
        os.makedirs(os.path.join(LOCAL_EVAL_DIR, corruption_name), exist_ok=True)

    # Corruptions are random, so a file that exists is never redone. Only missing files are generated.
    existing = {name: set(os.listdir(os.path.join(LOCAL_EVAL_DIR, name))) for name in CORRUPTIONS}
    pending = [p for p in image_paths if any(os.path.basename(p) not in existing[n] for n in CORRUPTIONS)]
    if not pending:
        print("All corruptions already exist. Nothing to generate.")
        raise SystemExit(0)

    read_flag = cv2.IMREAD_COLOR if args.dataset == 1 else cv2.IMREAD_UNCHANGED
    for img_path in tqdm(pending, desc="Applying Corruptions"):
        filename = os.path.basename(img_path)
        clean_img = cv2.imread(img_path, read_flag)
        if clean_img is None:
            continue

        for corruption_name, corruption_func in CORRUPTIONS.items():
            if filename not in existing[corruption_name]:
                cv2.imwrite(os.path.join(LOCAL_EVAL_DIR, corruption_name, filename), corruption_func(clean_img))

    print(f"\nAll corrupted datasets generated successfully on local SSD for {DATASET_NAME}!")