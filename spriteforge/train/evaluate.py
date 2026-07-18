# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Evaluation routines: PSNR, L1, codebook usage metrics, and qualitative review grids (Phase 5).
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spriteforge.model.config import get_config
from spriteforge.model.vqgan import SpriteVQGAN


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio between prediction and target."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 20.0 * np.log10(max_val / np.sqrt(mse))


def _local_variance(rgb: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Per-pixel local variance over a ksize x ksize window, summed across channels.
    Vectorized via box-filter moments (E[x^2] - E[x]^2), not a per-pixel Python loop.
    rgb: (H, W, C) float32 in [0, 1]. Returns (H, W) float32.
    """
    mean = cv2.blur(rgb, (ksize, ksize))
    mean_sq = cv2.blur(rgb * rgb, (ksize, ksize))
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    if var.ndim == 3:
        var = var.sum(axis=-1)
    return var


def compute_speckle_score(pred_rgba: np.ndarray, target_rgba: np.ndarray, flat_thresh: float = 0.001) -> float:
    """Speckle metric (see devlog/2026-07-08-random-pool-samples.md, diagnostic D1/D2): for
    every pixel whose ground-truth 3x3 neighborhood is nearly flat (real sprites are mostly
    flat color regions), measure how much local variance the *prediction* has there. A model
    with no speckle problem should have ~zero predicted variance wherever the ground truth is
    flat; rainbow/salt-and-pepper corruption shows up as high local variance exactly there.

    Higher = more speckle. Only RGB channels are considered (alpha excluded — matte softness
    is a different failure mode from color speckle).
    """
    gt_rgb = target_rgba[..., :3]
    pred_rgb = pred_rgba[..., :3]
    gt_var = _local_variance(gt_rgb)
    flat_mask = gt_var < flat_thresh
    if not np.any(flat_mask):
        return 0.0
    pred_var = _local_variance(pred_rgb)
    return float(pred_var[flat_mask].mean())


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    test_loader: DataLoader,
    device: str = "auto",
    output_grid_path: str | Path | None = None,
    num_grid_samples: int = 24,
    grid_columns: int = 8,
    sample_mode: str = "diverse",
    seed: int = 42,
    max_eval_samples: int | None = None
) -> dict[str, float]:
    """Compute quantitative metrics (PSNR, L1, codebook utilization) on held-out dataset."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = device

    print(f"[*] Loading checkpoint {ckpt_path} onto {device_str.upper()}...")
    ckpt = torch.load(ckpt_path, map_location=device_str, weights_only=False)
    
    config_name = ckpt.get("config_name", "size_32")
    config = get_config(config_name)
    
    model = SpriteVQGAN(config).to(device_str)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_psnr = 0.0
    total_l1 = 0.0
    total_speckle = 0.0
    unique_codes = set()
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device_str), targets.to(device_str)
            preds, _, indices = model(inputs)

            batch_size = inputs.shape[0]
            for i in range(batch_size):
                total_psnr += compute_psnr(preds[i], targets[i])
                total_l1 += F.l1_loss(preds[i], targets[i]).item()
                pred_np = preds[i].detach().cpu().numpy().transpose(1, 2, 0)
                target_np = targets[i].detach().cpu().numpy().transpose(1, 2, 0)
                total_speckle += compute_speckle_score(pred_np, target_np)
                num_samples += 1

            unique_codes.update(indices.view(-1).cpu().numpy().tolist())

            if max_eval_samples and num_samples >= max_eval_samples:
                break

    avg_psnr = total_psnr / max(1, num_samples)
    avg_l1 = total_l1 / max(1, num_samples)
    avg_speckle = total_speckle / max(1, num_samples)
    codebook_usage = len(unique_codes) / config.codebook_size

    metrics = {
        "psnr": round(avg_psnr, 2),
        "l1_loss": round(avg_l1, 4),
        "speckle_score": round(avg_speckle, 6),
        "codebook_usage": round(codebook_usage, 4),
        "active_codes": len(unique_codes),
        "total_codes": config.codebook_size
    }

    if output_grid_path:
        grid_inputs = []
        grid_preds = []
        grid_targets = []
        
        if hasattr(test_loader, "dataset") and len(test_loader.dataset) > 0:
            dataset = test_loader.dataset
            n_total = len(dataset)
            n_samples = min(num_grid_samples, n_total)
            
            if sample_mode == "diverse":
                indices = np.linspace(0, n_total - 1, n_samples, dtype=int)
            elif sample_mode == "random":
                np.random.seed(seed)
                indices = np.random.choice(n_total, size=n_samples, replace=False)
            else:
                indices = np.arange(n_samples)
                
            with torch.no_grad():
                for idx in indices:
                    inp, tgt = dataset[idx]
                    inp_dev = inp.unsqueeze(0).to(device_str)
                    pred, _, _ = model(inp_dev)
                    grid_inputs.append(inp.numpy())
                    grid_preds.append(pred[0].cpu().numpy())
                    grid_targets.append(tgt.numpy())
        
        if grid_inputs:
            blocks = []
            num_rows = (len(grid_inputs) + grid_columns - 1) // grid_columns
            for r in range(num_rows):
                sub_in = grid_inputs[r*grid_columns : (r+1)*grid_columns]
                sub_pr = grid_preds[r*grid_columns : (r+1)*grid_columns]
                sub_tg = grid_targets[r*grid_columns : (r+1)*grid_columns]
                
                while len(sub_in) < grid_columns:
                    sub_in.append(np.zeros_like(sub_in[0]))
                    sub_pr.append(np.zeros_like(sub_pr[0]))
                    sub_tg.append(np.zeros_like(sub_tg[0]))
                    
                in_row = np.concatenate([sub_in[i].transpose(1, 2, 0) for i in range(grid_columns)], axis=1)
                pr_row = np.concatenate([sub_pr[i].transpose(1, 2, 0) for i in range(grid_columns)], axis=1)
                tg_row = np.concatenate([sub_tg[i].transpose(1, 2, 0) for i in range(grid_columns)], axis=1)
                
                blocks.append(np.concatenate([in_row, pr_row, tg_row], axis=0))
                
            full_grid = np.concatenate(blocks, axis=0)
            full_grid_u8 = np.clip(full_grid * 255.0 + 0.5, 0, 255).astype(np.uint8)
            
            out_p = Path(output_grid_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(full_grid_u8, mode="RGBA").save(out_p)
            print(f"[+] Saved evaluation comparison grid ({len(grid_inputs)} samples, mode={sample_mode}) to {out_p}")

    return metrics
