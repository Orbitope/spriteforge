#!/bin/bash
# Sequentially trains the VQ-GAN on each per-source dataset split (100 epochs each,
# with a ramped adversarial warmup — see disc_start_epoch/adv_ramp_epochs in train.py),
# then evaluates the resulting checkpoint against that source's held-out test set.
# Run from the repo root. Intended to run unattended in the background.
# NOTE: written for bash 3.2 (macOS default) — no associative arrays.
set -uo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python

train_dir_for() {
  case "$1" in
    pmd) echo "data_private/train_32_pmd" ;;
    papi) echo "data_private/train_32_papi" ;;
    lpc) echo "data_private/train_32_lpc" ;;
    fe) echo "data_private/train_32_fe" ;;
  esac
}

test_dir_for() {
  case "$1" in
    pmd) echo "data_private/test_pmd_32_clean" ;;
    papi) echo "data_private/test_papi_32" ;;
    lpc) echo "data_private/test_lpc_32" ;;
    fe) echo "data_private/test_fe_32_clean" ;;
  esac
}

EPOCHS=100
DISC_START=10
ADV_RAMP_EPOCHS=20
mkdir -p checkpoints_bysource
SUMMARY_LOG="checkpoints_bysource/run_status.log"
echo "[run started] $(date)" > "$SUMMARY_LOG"

for source in pmd papi lpc fe; do
  train_dir="$(train_dir_for "$source")"
  test_dir="$(test_dir_for "$source")"
  out_dir="checkpoints_bysource/${source}"
  mkdir -p "$out_dir"
  echo "=== [$(date)] Training source: ${source} (${EPOCHS} epochs) ===" | tee -a "$SUMMARY_LOG"

  "$PY" -m spriteforge.cli train \
    --data-dir "$train_dir" \
    --size 32 \
    --epochs "$EPOCHS" \
    --output-dir "$out_dir" \
    --sample-interval 10 \
    --disc-start "$DISC_START" \
    --adv-ramp-epochs "$ADV_RAMP_EPOCHS" \
    > "${out_dir}/train_stdout.log" 2>&1
  train_status=$?

  if [ $train_status -ne 0 ]; then
    echo "[FAILED] training for ${source}, exit code ${train_status}. See ${out_dir}/train_stdout.log" | tee -a "$SUMMARY_LOG"
    continue
  fi
  echo "[OK] training for ${source} complete" | tee -a "$SUMMARY_LOG"

  ckpt="${out_dir}/vqgan_32_epoch_100.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[FAILED] checkpoint not found: ${ckpt}" | tee -a "$SUMMARY_LOG"
    continue
  fi

  echo "=== [$(date)] Evaluating source: ${source} ===" | tee -a "$SUMMARY_LOG"
  "$PY" -m spriteforge.cli eval \
    --checkpoint "$ckpt" \
    --data-dir "$test_dir" \
    -o "${out_dir}/eval_grid.png" \
    -s 32 \
    --batch-size 20 \
    --num-samples 20 \
    --sample-mode diverse \
    > "${out_dir}/eval_stdout.log" 2>&1
  eval_status=$?

  if [ $eval_status -ne 0 ]; then
    echo "[FAILED] eval for ${source}, exit code ${eval_status}. See ${out_dir}/eval_stdout.log" | tee -a "$SUMMARY_LOG"
  else
    echo "[OK] eval for ${source} complete" | tee -a "$SUMMARY_LOG"
  fi
done

echo "=== [$(date)] Generating review document ===" | tee -a "$SUMMARY_LOG"
"$PY" scripts/make_review_doc.py | tee -a "$SUMMARY_LOG"

echo "[run finished] $(date)" | tee -a "$SUMMARY_LOG"
