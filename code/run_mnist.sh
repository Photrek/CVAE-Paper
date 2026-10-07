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
NUM_SAMPLES=5
TARGET_DRIVE_DIR="/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/mnist_data"

source ./pipeline_lib.sh

KAPPAS=("0.0" "1e-6" "1e-4" "1e-2" "1e0" "1e2" "1e4" "1e6")

# =============================================================================
# PHASE 1: MODEL TRAINING (Mode 1, Dim=100)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 1: Training 100D Models..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Training 100D model for Parameter: $KAPPA"
    run_mode 1 "$KAPPA" $LATENT_DIM
done

# =============================================================================
# PHASE 2: GENERATE EVALUATION DATASETS (Mode 2, Dim=100)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 2: Generating Evaluation Sets..."
echo "------------------------------------------"
restore_eval_set
for KAPPA in "${KAPPAS[@]}"; do
    echo "Generating evaluation set for Parameter: $KAPPA"
    run_mode 2 "$KAPPA" $LATENT_DIM
done

# =============================================================================
# PHASE 3: CALCULATE METRICS & GENERATE CORRUPTIONS
# =============================================================================
echo "------------------------------------------"
echo "PHASE 3: Running Post-Processing Scripts..."
echo "------------------------------------------"
if python pipeline_state.py metrics ${DATASET_CHOICE} ${MODEL_CHOICE} ${LATENT_DIM} ${NUM_SAMPLES}; then
    echo "Standard metrics already computed. Skipping."
else
    python calculate_metrics.py ${DATASET_CHOICE} ${MODEL_CHOICE}
fi
if python pipeline_state.py corrupt ${DATASET_CHOICE}; then
    echo "Corruptions already exist. Skipping."
else
    python generate_corrupted_images.py ${DATASET_CHOICE}
fi

# Save originals + corruptions to Drive right away. The corruptions are random, so every model
# and every session must reuse this one set. Reconstructions are left out (they are rebuilt from the checkpoints).
echo "Saving originals and corruptions to Google Drive..."
sync_eval_zip no_recon

# =============================================================================
# PHASE 4: ROBUSTNESS INFERENCE (Mode 4, Dim=100)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 4: Generating Corrupted Reconstructions..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Running robustness inference for Parameter: $KAPPA"
    run_mode 4 "$KAPPA" $LATENT_DIM
done

# =============================================================================
# PHASE 5: RUN SUMMARY ROBUSTNESS METRICS & ZIP TO DRIVE
# =============================================================================
echo "------------------------------------------"
echo "PHASE 5: Aggregating Robustness Metrics..."
echo "------------------------------------------"
if python pipeline_state.py metrics ${DATASET_CHOICE} ${MODEL_CHOICE} ${LATENT_DIM} ${NUM_SAMPLES} robustness; then
    echo "Robustness metrics already computed. Skipping."
else
    python calculate_robustness_metrics.py ${DATASET_CHOICE} ${MODEL_CHOICE}
fi

sync_eval_zip

# =============================================================================
# PHASE 6: STOCHASTIC CONSISTENCY (Mode 5, Dim=100)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 6: Running Stochastic Consistency Tests..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Testing stochastic consistency for Parameter: $KAPPA"
    run_mode 5 "$KAPPA" $LATENT_DIM
done

# =============================================================================
# PHASE 7: STANDARD FREE ENERGY EVALUATION (Mode 7, Dim=100)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 7: Calculating Standard Free Energy (kappa=0 metric)..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Evaluating Standard Free Energy for Parameter: $KAPPA"
    run_mode 7 "$KAPPA" $LATENT_DIM
done

# =============================================================================
# PHASE 8: TRAIN 2D LATENT MODELS (Mode 1, Dim=2)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 8: Training 2D Models for Latent Space Analysis..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Training 2D model for Parameter: $KAPPA"
    run_mode 1 "$KAPPA" 2
done

# =============================================================================
# PHASE 9: GENERATE 2D LATENT MAPS / t-SNE PLOTS (Mode 6, Dim=2)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 9: Generating 2D Latent Maps / t-SNE Plots..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Generating 2D plot for Parameter: $KAPPA"
    run_mode 6 "$KAPPA" 2
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