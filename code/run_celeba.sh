#!/bin/bash

# Navigate to the code folder in Google Shared Drive (data folders stay at the project root)
cd "/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/code"

echo "=========================================="
echo "STARTING AUTOMATED CELEBA PIPELINE"
echo "=========================================="

# =============================================================================
# CONFIGURATION
# =============================================================================
# DATASET_CHOICE:
#   1 = CelebA (128x128 RGB)
#   2 = MNIST (32x32 Grayscale)
DATASET_CHOICE=1  

# MODEL_CHOICE:
#   1 = CVAE (Heavy-Tail MCG + Independent-Equals Scaling)
#   2 = Heavy-Tail VAE (Heavy-Tail MCG, NO Scaling)
#   3 = Beta-VAE (Standard Gaussian + Beta KL Weight)
#   4 = Prior-VAE (Standard Gaussian + Prior Variance 1/sqrt(beta))
MODEL_CHOICE=1    

LATENT_DIM=100
NUM_SAMPLES=5
TARGET_DRIVE_DIR="/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/celeba_data"

source ./pipeline_lib.sh

KAPPAS=("0" "1e-6" "1e-4" "1e-2" "1e0" "1e1")

# Start the slow Drive restore now, so it overlaps with phase 1. Skip phases 2-5 when they have nothing left to do.
if eval_phases_needed; then
    SKIP_EVAL=0
    start_restore_eval_set
else
    SKIP_EVAL=1
    echo "All models trained and all metrics computed. Phases 2-5 will be skipped."
fi

# =============================================================================
# PHASE 1: MODEL TRAINING (Mode 1)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 1: Training Models..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Training model for Parameter: $KAPPA"
    run_mode 1 "$KAPPA" $LATENT_DIM
done

# =============================================================================
# PHASE 2: GENERATE EVALUATION DATASETS (Mode 2)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 2: Generating Evaluation Sets..."
echo "------------------------------------------"
wait_restore_eval_set
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
if [ "$SKIP_EVAL" = 1 ] || python pipeline_state.py metrics ${DATASET_CHOICE} ${MODEL_CHOICE} ${LATENT_DIM} ${NUM_SAMPLES}; then
    echo "Standard metrics already computed. Skipping."
else
    python calculate_metrics.py ${DATASET_CHOICE} ${MODEL_CHOICE}
fi
if [ "$SKIP_EVAL" = 1 ] || python pipeline_state.py corrupt ${DATASET_CHOICE}; then
    echo "Corruptions already exist. Skipping."
else
    python generate_corrupted_images.py ${DATASET_CHOICE}
fi

# Save originals + corruptions to Drive right away. The corruptions are random, so every model
# and every session must reuse this one set. Reconstructions are left out (they are rebuilt from the checkpoints).
echo "Saving originals and corruptions to Google Drive..."
sync_eval_zip no_recon

# =============================================================================
# PHASE 4: ROBUSTNESS INFERENCE (Mode 4)
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
if [ "$SKIP_EVAL" = 1 ] || python pipeline_state.py metrics ${DATASET_CHOICE} ${MODEL_CHOICE} ${LATENT_DIM} ${NUM_SAMPLES} robustness; then
    echo "Robustness metrics already computed. Skipping."
else
    python calculate_robustness_metrics.py ${DATASET_CHOICE} ${MODEL_CHOICE}
fi

sync_eval_zip

# =============================================================================
# PHASE 6: STOCHASTIC CONSISTENCY (Mode 5)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 6: Running Stochastic Consistency Tests..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Testing stochastic consistency for Parameter: $KAPPA"
    run_mode 5 "$KAPPA" $LATENT_DIM
done

# =============================================================================
# PHASE 7: STANDARD FREE ENERGY EVALUATION (Mode 7)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 7: Calculating Standard Free Energy (kappa=0 metric)..."
echo "------------------------------------------"
for KAPPA in "${KAPPAS[@]}"; do
    echo "Evaluating Standard Free Energy for Parameter: $KAPPA"
    run_mode 7 "$KAPPA" $LATENT_DIM
done

echo "=========================================="
echo "CELEBA PIPELINE EXECUTION COMPLETED SUCCESSFULLY"
echo "=========================================="