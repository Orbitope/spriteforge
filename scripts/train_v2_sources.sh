#!/bin/bash
# Trains the VQ-GAN on the three bug-fixed sources (papi/lpc/fe, see
# devlog/2026-07-08-scraper-fixes-and-new-source.md) plus the new oga (OpenGameArt
# furniture/plants/environment) source, 100 epochs each, same hyperparameters validated in
# the first training round. Separate output dir (checkpoints_bysource_v2/) so the original
# pmd result (unaffected by any of these fixes) isn't touched or re-run.
set -uo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python

train_dir_for() {
  case "$1" in
    papi) echo "data_private/train_32_papi" ;;
    lpc) echo "data_private/train_32_lpc" ;;
    fe) echo "data_private/train_32_fe" ;;
    sr) echo "data_private/train_32_sr_combined" ;;
  esac
}

test_dir_for() {
  case "$1" in
    papi) echo "data_private/test_papi_32_v2" ;;
    lpc) echo "data_private/test_lpc_32_v2" ;;
    fe) echo "data_private/test_fe_32_v2" ;;
    sr) echo "data_private/test_sr_32" ;;
  esac
}

EPOCHS=100
DISC_START=10
ADV_RAMP_EPOCHS=20
mkdir -p checkpoints_bysource_v2
SUMMARY_LOG="checkpoints_bysource_v2/run_status.log"
echo "[run started] $(date)" > "$SUMMARY_LOG"

# oga dropped 2026-07-08 (932 images, speckle 68x worse than fe); replaced by sr.
# sr = Spriters Resource, 9,674 humanoid sprites from 6 SNES/GBA/DS games (FF6, Chrono
# Trigger, Zelda ALTTP, Mega Man X, Castlevania AoS, Castlevania DoS). Scraped 2026-07-08.
# NOTE: sr test split must be created before running — see scripts/split_sr_test.sh
for source in papi lpc fe sr; do
  train_dir="$(train_dir_for "$source")"
  test_dir="$(test_dir_for "$source")"
  out_dir="checkpoints_bysource_v2/${source}"
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

echo "[run finished] $(date)" | tee -a "$SUMMARY_LOG"
