#!/bin/bash
# Train VQ-GAN on SR dataset variants:
#   1. sr (combined) — all 25,958 sprites
#   2. sr_cvdos — Castlevania DoS (8,491 sprites)
#   3. sr_ct — Chrono Trigger (3,598 sprites)
#   4. sr_som — Secret of Mana (3,021 sprites)
# Each run: 100 epochs, same hyperparameters as v2.

set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

train_dir_for() {
  case "$1" in
    sr) echo "data_private/train_32_sr_combined" ;;
    sr_cvdos) echo "data_private/train_32_sr_cvdos" ;;
    sr_ct) echo "data_private/train_32_sr_ct" ;;
    sr_som) echo "data_private/train_32_sr_som" ;;
  esac
}

test_dir_for() {
  case "$1" in
    sr) echo "data_private/test_sr_32" ;;
    sr_cvdos) echo "data_private/test_sr_cvdos_32" ;;
    sr_ct) echo "data_private/test_sr_ct_32" ;;
    sr_som) echo "data_private/test_sr_som_32" ;;
  esac
}

EPOCHS=100
DISC_START=10
ADV_RAMP_EPOCHS=20
mkdir -p checkpoints_sr_variants
SUMMARY_LOG="checkpoints_sr_variants/run_status.log"
echo "[run started] $(date)" > "$SUMMARY_LOG"
echo "Training 4 SR variants: combined, CV:DoS, Chrono Trigger, Secret of Mana" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"

for source in sr sr_cvdos sr_ct sr_som; do
  train_dir="$(train_dir_for "$source")"
  test_dir="$(test_dir_for "$source")"
  out_dir="checkpoints_sr_variants/${source}"
  mkdir -p "$out_dir"

  num_files=$(ls "$train_dir" | wc -l)
  echo "=== [$(date)] Training: ${source} (${num_files} sprites, ${EPOCHS} epochs) ===" | tee -a "$SUMMARY_LOG"

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

  echo "=== [$(date)] Evaluating: ${source} ===" | tee -a "$SUMMARY_LOG"
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

  echo "" >> "$SUMMARY_LOG"
done

echo "[run finished] $(date)" | tee -a "$SUMMARY_LOG"
