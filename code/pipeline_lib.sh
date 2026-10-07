#!/bin/bash
# Helpers shared by run_celeba.sh and run_mnist.sh. Source it after setting
# DATASET_CHOICE, MODEL_CHOICE, NUM_SAMPLES and TARGET_DRIVE_DIR.
# Every helper skips work that is already finished, so a resumed Colab run does not burn GPU credits.
# The checks live in pipeline_state.py (standard library only, so they take milliseconds).

LOCAL_EVAL_DIR="/content/evaluation_dataset"
EVAL_ZIP="${TARGET_DRIVE_DIR}/evaluation_dataset.zip"

# run_mode MODE KAPPA DIM: runs run.py for one kappa unless that step is already done.
run_mode() {
    local mode=$1 kappa=$2 dim=$3
    if python pipeline_state.py step "$DATASET_CHOICE" "$MODEL_CHOICE" "$kappa" "$dim" "$NUM_SAMPLES" "$mode"; then
        echo "  Already done for Parameter $kappa (mode $mode). Skipping."
    else
        printf "${DATASET_CHOICE}\n${MODEL_CHOICE}\n${kappa}\n${dim}\n${NUM_SAMPLES}\n${mode}\n" | python run.py
    fi
}

# restore_eval_set: after a Colab reset, unpack the shared eval set (originals, corruptions, saved reconstructions) once.
restore_eval_set() {
    if [ ! -d "${LOCAL_EVAL_DIR}/originals" ] && [ -f "$EVAL_ZIP" ]; then
        echo "Restoring the evaluation set from Drive..."
        unzip -q -o "$EVAL_ZIP" -d /content
    fi
}

# sync_eval_zip [no_recon]: writes the eval set to Drive only when a local file is newer than the last sync.
# zip -u adds the new files and keeps the old ones, so nothing is compressed twice.
# The full sync (reconstructions included) keeps its own stamp file, because reconstructions made before the
# no_recon zip are older than the zip but still have to go in once.
sync_eval_zip() {
    local prune=() ref="$EVAL_ZIP"
    if [ "$1" = "no_recon" ]; then
        prune=(-path "evaluation_dataset/reconstructions*" -prune -o)
    else
        ref="${EVAL_ZIP}.full_sync"
    fi
    if [ -f "$EVAL_ZIP" ] && [ -f "$ref" ] && [ -z "$(cd /content && find evaluation_dataset "${prune[@]}" -type f -newer "$ref" -print -quit)" ]; then
        echo "Drive zip is up to date. Skipping."
        return
    fi
    echo "Updating the evaluation zip on Google Drive..."
    if [ "$1" = "no_recon" ]; then
        (cd /content && zip -r -u -q "$EVAL_ZIP" evaluation_dataset -x "evaluation_dataset/reconstructions*")
    else
        (cd /content && zip -r -u -q "$EVAL_ZIP" evaluation_dataset) && touch "$ref"
    fi
}
