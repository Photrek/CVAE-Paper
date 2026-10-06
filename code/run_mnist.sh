#!/bin/bash

# Navigate to the code folder in Google Shared Drive (data folders stay at the project root)
cd "/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/code"

echo "=========================================="
echo "STARTING FULL AUTOMATED MNIST PIPELINE"
echo "=========================================="

# =============================================================================
# CONFIGURATION
# =============================================================================
# DATASET_CHOICE:
#   1 = CelebA (128x128 RGB)
#   2 = MNIST (32x32 Grayscale)
DATASET_CHOICE=2   

# MODEL_CHOICE:
#   1 = CVAE (Heavy-Tail MCG + Independent-Equals Scaling)
#   2 = Heavy-Tail VAE (Heavy-Tail MCG, NO Scaling)
#   3 = Beta-VAE (Standard Gaussian + Beta KL Weight)
#   4 = Prior-VAE (Standard Gaussian + Prior Variance 1/sqrt(beta))
MODEL_CHOICE=1     

LATENT_DIM=100
NUM_SAMPLES=1
TARGET_DRIVE_DIR="/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/mnist_data"

KAPPAS=("0.0" "1e-6" "1e-4" "1e-2" "1e0" "1e2" "1e4" "1e6")

# =============================================================================
# PHASE 1: MODEL TRAINING (Mode 1, Dim=100)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 1: Training 100D Models..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Training 100D model for Parameter: $KAPPA"
    printf "${DATASET_CHOICE}\n${MODEL_CHOICE}\n${KAPPA}\n${LATENT_DIM}\n${NUM_SAMPLES}\n1\n" | python run.py
done

# =============================================================================
# PHASE 2: GENERATE EVALUATION DATASETS (Mode 2, Dim=100)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 2: Generating Evaluation Sets..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Generating evaluation set for Parameter: $KAPPA"
    printf "${DATASET_CHOICE}\n${MODEL_CHOICE}\n${KAPPA}\n${LATENT_DIM}\n${NUM_SAMPLES}\n2\n" | python run.py
done

# =============================================================================
# PHASE 3: CALCULATE METRICS & GENERATE CORRUPTIONS
# =============================================================================
echo "------------------------------------------"
echo "PHASE 3: Running Post-Processing Scripts..."
echo "------------------------------------------"
python calculate_metrics.py ${DATASET_CHOICE}
python generate_corrupted_images.py ${DATASET_CHOICE}

# =============================================================================
# PHASE 4: ROBUSTNESS INFERENCE (Mode 4, Dim=100)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 4: Generating Corrupted Reconstructions..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Running robustness inference for Parameter: $KAPPA"
    printf "${DATASET_CHOICE}\n${MODEL_CHOICE}\n${KAPPA}\n${LATENT_DIM}\n${NUM_SAMPLES}\n4\n" | python run.py
done

# =============================================================================
# PHASE 5: RUN SUMMARY ROBUSTNESS METRICS & ZIP TO DRIVE
# =============================================================================
echo "------------------------------------------"
echo "PHASE 5: Aggregating Robustness Metrics..."
echo "------------------------------------------"
python calculate_robustness_metrics.py ${DATASET_CHOICE}

echo "Zipping local evaluation dataset to Google Drive..."
zip -r -q "${TARGET_DRIVE_DIR}/evaluation_dataset.zip" /content/evaluation_dataset

# =============================================================================
# PHASE 6: STOCHASTIC CONSISTENCY (Mode 5, Dim=100)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 6: Running Stochastic Consistency Tests..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Testing stochastic consistency for Parameter: $KAPPA"
    printf "${DATASET_CHOICE}\n${MODEL_CHOICE}\n${KAPPA}\n${LATENT_DIM}\n${NUM_SAMPLES}\n5\n" | python run.py
done

# =============================================================================
# PHASE 7: STANDARD FREE ENERGY EVALUATION (Mode 7, Dim=100)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 7: Calculating Standard Free Energy (kappa=0 metric)..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Evaluating Standard Free Energy for Parameter: $KAPPA"
    printf "${DATASET_CHOICE}\n${MODEL_CHOICE}\n${KAPPA}\n${LATENT_DIM}\n${NUM_SAMPLES}\n7\n" | python run.py
done

# =============================================================================
# PHASE 8: TRAIN 2D LATENT MODELS (Mode 1, Dim=2)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 8: Training 2D Models for Latent Space Analysis..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Training 2D model for Parameter: $KAPPA"
    printf "${DATASET_CHOICE}\n${MODEL_CHOICE}\n${KAPPA}\n2\n${NUM_SAMPLES}\n1\n" | python run.py
done

# =============================================================================
# PHASE 9: GENERATE 2D LATENT MAPS / t-SNE PLOTS (Mode 6, Dim=2)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 9: Generating 2D Latent Maps / t-SNE Plots..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Generating 2D plot for Parameter: $KAPPA"
    printf "${DATASET_CHOICE}\n${MODEL_CHOICE}\n${KAPPA}\n2\n${NUM_SAMPLES}\n6\n" | python run.py
done

# =============================================================================
# PHASE 10: DATASET CLEANUP (Mode 3)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 10: Cleaning Up Extracted Uncompressed Data..."
echo "------------------------------------------"
printf "${DATASET_CHOICE}\n${MODEL_CHOICE}\n0.0\n${LATENT_DIM}\n${NUM_SAMPLES}\n3\n" | python run.py

echo "=========================================="
echo "MNIST PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="