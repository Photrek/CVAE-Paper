#!/bin/bash
# Helpers for run_pipeline.sh. Source it after setting
# DATASET_CHOICE, MODEL_CHOICE, NUM_SAMPLES and TARGET_DRIVE_DIR.
# Every helper skips work that is already finished, so a resumed Colab run does not burn GPU credits.
# The checks live in pipeline_state.py (standard library only, so they take milliseconds).

LOCAL_EVAL_DIR="/content/evaluation_dataset"
EVAL_ZIP="${TARGET_DRIVE_DIR}/evaluation_dataset.zip"

# run_mode MODE KAPPA DIM: runs run.py for one kappa unless that step is already done.
run_mode() {
    local mode=$1 kappa=$2 dim=$3
    if [ "$SKIP_EVAL" = 1 ] && { [ "$mode" = 2 ] || [ "$mode" = 4 ]; }; then
        echo "  Metrics already computed. Skipping mode $mode for Parameter $kappa."
        return
    fi
    if [ "$mode" != 1 ] && python pipeline_state.py diverged "$DATASET_CHOICE" "$MODEL_CHOICE" "$kappa" "$dim" "$NUM_SAMPLES"; then
        echo "  Training diverged (nan row in the epoch log) for Parameter $kappa. Skipping mode $mode."
        return
    fi
    if python pipeline_state.py step "$DATASET_CHOICE" "$MODEL_CHOICE" "$kappa" "$dim" "$NUM_SAMPLES" "$mode"; then
        echo "  Already done for Parameter $kappa (mode $mode). Skipping."
    else
        printf "${DATASET_CHOICE}\n${MODEL_CHOICE}\n${kappa}\n${dim}\n${NUM_SAMPLES}\n${mode}\n" | python run.py
    fi
}

# eval_phases_needed: false when every model is trained and both metric tables are complete, which means phases 2-5
# have nothing left to do and the eval set does not have to be restored. Needs PARAMS, LATENT_DIM and NUM_SAMPLES.
eval_phases_needed() {
    local k
    for k in "${PARAMS[@]}"; do
        python pipeline_state.py step "$DATASET_CHOICE" "$MODEL_CHOICE" "$k" "$LATENT_DIM" "$NUM_SAMPLES" 1 || return 0
    done
    python pipeline_state.py metrics "$DATASET_CHOICE" "$MODEL_CHOICE" "$LATENT_DIM" "$NUM_SAMPLES" || return 0
    python pipeline_state.py metrics "$DATASET_CHOICE" "$MODEL_CHOICE" "$LATENT_DIM" "$NUM_SAMPLES" robustness || return 0
    return 1
}

# start_restore_eval_set: after a Colab reset, copies the shared eval zip (originals, corruptions, saved reconstructions)
# to the local SSD and unpacks it in the background, while phase 1 uses the GPU. wait_restore_eval_set joins it.
# Reading one big file in a row from Drive is faster than unzipping straight off the Drive mount.
start_restore_eval_set() {
    RESTORE_PID=""
    if [ ! -d "${LOCAL_EVAL_DIR}/originals" ] && [ -f "$EVAL_ZIP" ]; then
        echo "Restoring the evaluation set from Drive in the background..."
        (
            start=$SECONDS
            cp "$EVAL_ZIP" /content/eval_restore.zip && unzip -q -o /content/eval_restore.zip -d /content
            rm -f /content/eval_restore.zip
            echo "Evaluation set restored in $((SECONDS - start)) s."
        ) &
        RESTORE_PID=$!
    fi
}

wait_restore_eval_set() {
    [ -n "$RESTORE_PID" ] && wait "$RESTORE_PID"
    RESTORE_PID=""
}

# sync_eval_zip [no_recon]: writes the eval set to Drive only when a local file is newer than the last sync.
# zip -u adds the new files and keeps the old ones, so nothing is compressed twice.
# The full sync (reconstructions included) keeps its own stamp file, because reconstructions made before the
# no_recon zip are older than the zip but still have to go in once.
sync_eval_zip() {
    [ "$SKIP_EVAL" = 1 ] && return
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
