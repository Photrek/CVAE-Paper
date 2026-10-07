#!/bin/bash

# Navigate to the code folder in Google Shared Drive (data folders stay at the project root)
cd "/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper/code"

# =============================================================================
# CONFIGURATION
# =============================================================================
# Both choices come from the caller, so no edit is needed to switch:
#   DATASET_CHOICE=2 MODEL_CHOICE=3 ./run_pipeline.sh
# DATASET_CHOICE (default 1):
#   1 = CelebA (128x128 RGB)
#   2 = MNIST (32x32 Grayscale)
# MODEL_CHOICE (default 1):
#   1 = CVAE (Heavy-Tail MCG + Independent-Equals Scaling)
#   2 = Heavy-Tail VAE (Heavy-Tail MCG, NO Scaling)
#   3 = Beta-VAE (Standard Gaussian + Beta KL Weight)
#   4 = Prior-VAE (Standard Gaussian + Prior Variance 1/sqrt(beta))
DATASET_CHOICE=${DATASET_CHOICE:-1}
MODEL_CHOICE=${MODEL_CHOICE:-1}

LATENT_DIM=100
NUM_SAMPLES=5
PROJECT_DIR="/content/drive/Shareddrives/Photrek & its Partners/Projects/CVAE Paper"

BETAS=("1e-1" "5e-1" "1e0" "2e0" "5e0" "1e1")  # models 3-4 (Beta-VAE, Prior-VAE)
if [ "$DATASET_CHOICE" = 1 ]; then
    DATASET_NAME="CELEBA"
    TARGET_DRIVE_DIR="${PROJECT_DIR}/celeba_data"
    KAPPAS=("0" "1e-6" "1e-4" "1e-2" "1e0" "1e1")  # models 1-2 (CVAE, Heavy-Tail VAE)
elif [ "$DATASET_CHOICE" = 2 ]; then
    DATASET_NAME="MNIST"
    TARGET_DRIVE_DIR="${PROJECT_DIR}/mnist_data"
    KAPPAS=("0" "1e-6" "1e-4" "1e-2" "1e0" "1e2" "1e4" "1e6")
else
    echo "Unknown DATASET_CHOICE=${DATASET_CHOICE} (use 1 for CelebA, 2 for MNIST)."
    exit 1
fi

echo "=========================================="
echo "STARTING AUTOMATED ${DATASET_NAME} PIPELINE"
echo "=========================================="

source ./pipeline_lib.sh

# PARAMS is the grid the phases loop over. run.py reads it as kappa for models 1-2 and as beta for models 3-4.
if [ "$MODEL_CHOICE" = 3 ] || [ "$MODEL_CHOICE" = 4 ]; then
    PARAMS=("${BETAS[@]}")
else
    PARAMS=("${KAPPAS[@]}")
fi
echo "Model ${MODEL_CHOICE}, grid: ${PARAMS[*]}"

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
for PARAM in "${PARAMS[@]}"; do
    echo "Training model for Parameter: $PARAM"
    run_mode 1 "$PARAM" $LATENT_DIM
done

# =============================================================================
# PHASE 2: GENERATE EVALUATION DATASETS (Mode 2)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 2: Generating Evaluation Sets..."
echo "------------------------------------------"
wait_restore_eval_set
for PARAM in "${PARAMS[@]}"; do
    echo "Generating evaluation set for Parameter: $PARAM"
    run_mode 2 "$PARAM" $LATENT_DIM
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
for PARAM in "${PARAMS[@]}"; do
    echo "Running robustness inference for Parameter: $PARAM"
    run_mode 4 "$PARAM" $LATENT_DIM
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
for PARAM in "${PARAMS[@]}"; do
    echo "Testing stochastic consistency for Parameter: $PARAM"
    run_mode 5 "$PARAM" $LATENT_DIM
done

# =============================================================================
# PHASE 7: STANDARD FREE ENERGY EVALUATION (Mode 7)
# =============================================================================
echo "------------------------------------------"
echo "PHASE 7: Calculating Standard Free Energy (kappa=0 metric)..."
echo "------------------------------------------"
for PARAM in "${PARAMS[@]}"; do
    echo "Evaluating Standard Free Energy for Parameter: $PARAM"
    run_mode 7 "$PARAM" $LATENT_DIM
done

# =============================================================================
# PHASES 8-10 (MNIST only): 2D LATENT MODELS, LATENT MAPS, CLEANUP
# =============================================================================
if [ "$DATASET_CHOICE" = 2 ]; then
    echo "------------------------------------------"
    echo "PHASE 8: Training 2D Models for Latent Space Analysis..."
    echo "------------------------------------------"
    for PARAM in "${PARAMS[@]}"; do
        echo "Training 2D model for Parameter: $PARAM"
        run_mode 1 "$PARAM" 2
    done

    echo "------------------------------------------"
    echo "PHASE 9: Generating 2D Latent Maps / t-SNE Plots..."
    echo "------------------------------------------"
    for PARAM in "${PARAMS[@]}"; do
        echo "Generating 2D plot for Parameter: $PARAM"
        run_mode 6 "$PARAM" 2
    done

    echo "------------------------------------------"
    echo "PHASE 10: Cleaning Up Extracted Uncompressed Data..."
    echo "------------------------------------------"
    printf "${DATASET_CHOICE}\n${MODEL_CHOICE}\n${PARAMS[0]}\n${LATENT_DIM}\n${NUM_SAMPLES}\n3\n" | python run.py
fi

echo "=========================================="
echo "${DATASET_NAME} PIPELINE EXECUTION COMPLETED SUCCESSFULLY"
echo "=========================================="