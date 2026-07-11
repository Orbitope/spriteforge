import csv
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from spriteforge.model.config import get_config, ModelConfig
from spriteforge.model.discriminator import PatchDiscriminator
from spriteforge.model.losses import SpriteReconstructionLoss, OrthogonalRegularization
from spriteforge.train.evaluate import compute_speckle_score
from spriteforge.model.vqgan import SpriteVQGAN
from spriteforge.train.artifacts import save_sample_grid


def get_device(device_arg: str | None = None) -> str:
    """Automatically select best available computing device (CUDA, Apple Silicon MPS, or CPU)."""
    if device_arg and device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SpriteDataset(Dataset):
    """Loads pristine sprites from a directory and generates on-the-fly degraded inputs for Stage A training."""
    def __init__(self, data_dir: str | Path, target_size: int = 32, apply_degradation: bool = True, only_transparent: bool = True, degrade_profile: str = "legacy"):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.apply_degradation = apply_degradation
        self.only_transparent = only_transparent
        # "legacy": original standard/realistic/color_shift mix.
        # "ai_style": dominated by ai_style_source() — structure-preserving,
        #   background-injecting, AI-artifact-rich. See devlog 2026-07-09 domain-gap.
        self.degrade_profile = degrade_profile
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        self.image_paths = sorted(
            list(self.data_dir.glob("**/*.png")) +
            list(self.data_dir.glob("**/*.bmp")) +
            list(self.data_dir.glob("**/*.webp"))
        )
        if self.only_transparent:
            # Filter out known opaque datasets (like Fire Emblem GBA battle scenes starting with 'fe_')
            self.image_paths = [p for p in self.image_paths if not p.name.startswith("fe_")]

        if not self.image_paths:
            raise ValueError(f"No image files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        try:
            pil_img = Image.open(img_path).convert("RGBA")
            img_rgba = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception:
            # Fallback to empty canvas if file is corrupted
            img_rgba = np.zeros((self.target_size, self.target_size, 4), dtype=np.float32)

        # Ensure exact dimensions
        if img_rgba.shape[:2] != (self.target_size, self.target_size):
            from spriteforge.core.resize import pad_to_target
            padded = pad_to_target(img_rgba, target_size=self.target_size)
            img_rgba = padded if padded is not None else np.zeros((self.target_size, self.target_size, 4), dtype=np.float32)

        # Force transparent pixels to have black RGB to strip hidden chroma keys (like #FF00FF pink)
        # Otherwise the discriminator learns to look for invisible pink backgrounds!
        if img_rgba.shape[-1] == 4:
            img_rgba[..., :3] = img_rgba[..., :3] * img_rgba[..., 3:4]

        target_tensor = torch.from_numpy(img_rgba).permute(2, 0, 1).contiguous()

        if self.apply_degradation:
            from spriteforge.core.degrade import degrade, DegradeRanges
            if self.degrade_profile in ("ai_style", "ai_style_crisp"):
                # PoC profiles (devlog 2026-07-09): the real-input distribution SPANS
                # crisp near-sprite images (AI generated close to the target style) AND
                # mushy ones (complex source art that turns to mud when downscaled to 32).
                #   ai_style       — spans the range (crisp end + mush end + middle).
                #   ai_style_crisp — leans crisp (near-sprite specialist); A/B against
                #                    the spanning mix to see which real inputs each favors.
                if self.degrade_profile == "ai_style_crisp":
                    probs = [0.7, 0.2, 0.1]   # crisp-heavy
                else:
                    probs = [0.45, 0.2, 0.35]  # spanning
                mode = np.random.choice(["ai_style", "realistic", "standard"], p=probs)
                if mode == "ai_style":
                    ranges = DegradeRanges.ai_style_source()
                elif mode == "realistic":
                    ranges = DegradeRanges.realistic_low_noise()
                else:
                    ranges = DegradeRanges.standard()
            else:
                # Rebalanced toward harder cases (Stage 2, methodology review Part 1b):
                # was [0.2, 0.5, 0.3] which put 80% of training on mild presets and
                # buried the full-severity "standard" case the model fails hardest on.
                mode = np.random.choice(["standard", "realistic", "color_shift"], p=[0.45, 0.35, 0.2])
                if mode == "realistic":
                    ranges = DegradeRanges.realistic_low_noise()
                elif mode == "color_shift":
                    ranges = DegradeRanges.color_shift_only()
                else:
                    ranges = DegradeRanges.standard()
            # multiscale=True (Stage 2, Part 1a): codec/optics primitives run at
            # 4-8x upscale then area-downscale, so JPEG/blur/noise artifacts are
            # sized like real compress-then-downscale instead of spanning a
            # quarter of the 32x32 grid.
            degraded_rgba = degrade(img_rgba, ranges=ranges, multiscale=True)
            input_tensor = torch.from_numpy(degraded_rgba).permute(2, 0, 1).contiguous()
        else:
            input_tensor = target_tensor.clone()

        return input_tensor, target_tensor


def train_model(
    config_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 100,
    lr: float = 2e-4,
    device: str = "auto",
    output_dir: str | Path = "checkpoints",
    save_interval: int = 10,
    sample_interval: int = 25,
    disc_start_epoch: int = 3,
    adv_weight: float = 1.0,
    fm_weight: float = 0.2,
    adv_ramp_epochs: int = 10,
    resume_path: str | Path | None = None
) -> SpriteVQGAN:
    """Train VQ-GAN reconstruction model with adversarial warmup and feature matching losses."""
    device_str = get_device(device)
    print(f"[*] Initializing VQ-GAN training on device: {device_str.upper()} (GAN warmup: {disc_start_epoch} epochs)")

    config = get_config(config_name)
    model = SpriteVQGAN(config).to(device_str)
    critic = PatchDiscriminator(in_channels=config.out_channels).to(device_str)

    if resume_path:
        ckpt_path = Path(resume_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        print(f"[*] Resuming model weights from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device_str, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "critic_state_dict" in ckpt:
            try:
                critic.load_state_dict(ckpt["critic_state_dict"])
                print("[*] Successfully loaded critic state dict.")
            except Exception as e:
                print(f"[!] Could not load critic state dict: {e}")

    opt_g = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(critic.parameters(), lr=lr, betas=(0.5, 0.999))
    
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=1e-6)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-6)

    use_cuda_amp = (device_str == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    recon_loss_fn = SpriteReconstructionLoss().to(device_str)
    ortho_metric_fn = OrthogonalRegularization().to(device_str)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Phase 5 devlog: per-epoch CSV for dimensional-collapse tracking.
    csv_path = out_path / "training_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["epoch", "loss_g", "loss_d", "recon", "vq", "ortho", "active_codes", "total_codes", "speckle"]
    )

    print(f"[*] Starting training loop for {epochs} epochs (Dataset size: {len(train_loader.dataset)} sprites)...")
    start_time = time.time()
    best_recon = float("inf")

    try:
      for epoch in range(1, epochs + 1):
        model.train()
        critic.train()
        
        total_loss_g_epoch = 0.0
        total_loss_d_epoch = 0.0
        total_recon_epoch = 0.0
        total_vq_epoch = 0.0
        epoch_codes: set[int] = set()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device_str), targets.to(device_str)

            # --------------------------------------------------------------- #
            # 1. Train Critic (PatchDiscriminator) - only after warmup
            # --------------------------------------------------------------- #
            if epoch >= disc_start_epoch:
                opt_d.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device_str if device_str in ["cuda", "cpu", "mps"] else "cpu", enabled=use_cuda_amp):
                    with torch.no_grad():
                        preds, _, _ = model(inputs)
                    
                    real_logits = critic(targets)
                    fake_logits = critic(preds.detach())

                    # Hinge loss
                    loss_d_real = torch.mean(F.relu(1.0 - real_logits))
                    loss_d_fake = torch.mean(F.relu(1.0 + fake_logits))
                    loss_d = (loss_d_real + loss_d_fake) * 0.5

                if use_cuda_amp:
                    scaler.scale(loss_d).backward()
                    scaler.step(opt_d)
                else:
                    loss_d.backward()
                    opt_d.step()
            else:
                loss_d = torch.tensor(0.0, device=device_str)

            # --------------------------------------------------------------- #
            # 2. Train VQ-GAN Generator
            # --------------------------------------------------------------- #
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_str if device_str in ["cuda", "cpu", "mps"] else "cpu", enabled=use_cuda_amp):
                preds, vq_loss, indices = model(inputs)

                recon_loss = recon_loss_fn(preds, targets)
                
                if epoch >= disc_start_epoch:
                    # Adversarial loss
                    gen_logits = critic(preds)
                    loss_g_adv = -torch.mean(gen_logits)

                    # Feature matching loss
                    real_feats = critic.extract_features(targets)
                    fake_feats = critic.extract_features(preds)
                    loss_fm = sum(F.l1_loss(f_fake, f_real.detach()) for f_fake, f_real in zip(fake_feats, real_feats)) / len(real_feats)

                    # Linearly ramp adversarial/feature-matching weight from 0 -> target over
                    # adv_ramp_epochs, instead of switching to full weight instantly. An
                    # instant full-weight switch against a nearly-untrained critic destabilizes
                    # the generator (observed: recon loss regressed and never recovered).
                    ramp = min(1.0, (epoch - disc_start_epoch + 1) / max(1, adv_ramp_epochs))
                    total_loss_g = recon_loss + vq_loss + ramp * adv_weight * loss_g_adv + ramp * fm_weight * loss_fm
                else:
                    loss_g_adv = torch.tensor(0.0, device=device_str)
                    loss_fm = torch.tensor(0.0, device=device_str)
                    total_loss_g = recon_loss + vq_loss

            if use_cuda_amp:
                scaler.scale(total_loss_g).backward()
                scaler.step(opt_g)
                scaler.update()
            else:
                total_loss_g.backward()
                opt_g.step()

            total_loss_g_epoch += total_loss_g.item()
            total_loss_d_epoch += loss_d.item()
            total_recon_epoch += recon_loss.item()
            total_vq_epoch += vq_loss.item()
            epoch_codes.update(indices.detach().view(-1).cpu().tolist())

            # Speckle metric (see devlog/2026-07-08-random-pool-samples.md D1/D2): recon/vq
            # loss don't reliably reflect this failure mode, so track it directly. Only on
            # the first batch of each epoch — cv2-based, cheap per-call, but not worth paying
            # on every batch of a large dataset for a once-per-epoch trend signal.
            if batch_idx == 0:
                with torch.no_grad():
                    batch_speckle = 0.0
                    n_speckle = min(8, preds.shape[0])
                    for i in range(n_speckle):
                        pred_np = preds[i].detach().float().cpu().numpy().transpose(1, 2, 0)
                        target_np = targets[i].detach().float().cpu().numpy().transpose(1, 2, 0)
                        batch_speckle += compute_speckle_score(pred_np, target_np)
                    epoch_speckle = batch_speckle / max(1, n_speckle)

        num_batches = max(1, len(train_loader))
        avg_g = total_loss_g_epoch / num_batches
        avg_d = total_loss_d_epoch / num_batches
        avg_recon = total_recon_epoch / num_batches
        avg_vq = total_vq_epoch / num_batches
        elapsed = time.time() - start_time
        current_lr = scheduler_g.get_last_lr()[0]

        print(f"[Epoch {epoch:03d}/{epochs:03d}] ({elapsed:.1f}s, LR: {current_lr:.2e}) | "
              f"Loss G: {avg_g:.4f} (Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}) | Loss D: {avg_d:.4f}")

        with torch.no_grad():
            ortho_val = ortho_metric_fn(model.quantizer.embedding).item()
        active_codes = len(epoch_codes)
        csv_writer.writerow([
            epoch, round(avg_g, 4), round(avg_d, 4), round(avg_recon, 4),
            round(avg_vq, 4), round(ortho_val, 4), active_codes, config.codebook_size,
            round(epoch_speckle, 6),
        ])
        csv_file.flush()
        print(f"           ortho: {ortho_val:.4f} | active_codes: {active_codes}/{config.codebook_size}")

        if epoch % sample_interval == 0 or epoch == epochs:
            try:
                grid_path = save_sample_grid(model, train_loader.dataset, epoch, out_path / "samples", device_str)
                print(f"    [+] Saved devlog sample grid: {grid_path}")
            except Exception as e:
                print(f"    [!] Sample grid failed: {e}")

        scheduler_g.step()
        if epoch >= disc_start_epoch:
            scheduler_d.step()

        if epoch % save_interval == 0 or epoch == epochs:
            ckpt_name = f"vqgan_{config.target_size}_epoch_{epoch:03d}.pt"
            ckpt_path = out_path / ckpt_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "opt_g_state_dict": opt_g.state_dict(),
                "opt_d_state_dict": opt_d.state_dict(),
                "config_name": config_name,
            }, ckpt_path)
            print(f"    [+] Saved checkpoint: {ckpt_path}")

        # Track the best-reconstruction checkpoint separately, so a good early state
        # survives even if later adversarial training degrades reconstruction quality.
        if avg_recon < best_recon:
            best_recon = avg_recon
            best_path = out_path / f"vqgan_{config.target_size}_best_recon.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "opt_g_state_dict": opt_g.state_dict(),
                "opt_d_state_dict": opt_d.state_dict(),
                "config_name": config_name,
                "recon_loss": avg_recon,
            }, best_path)
    finally:
        csv_file.close()

    print(f"[+] Training completed successfully in {time.time() - start_time:.1f}s!")
    return model
