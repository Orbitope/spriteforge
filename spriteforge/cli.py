# Spriteforge — retro sprite restoration studio
# Copyright (C) 2026 Matthew Wesley Burke
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Spriteforge CLI entry point.

Supports:
- convert: Deterministic Stage A conversion (downscale -> OKLab palette snap -> hard alpha).
- calibrate: Generate side-by-side grids of real downscaled inputs vs synthetic degraded sprites.
- ingest: Slice and normalize sprite sheets into training datasets.
- scrape: Automated downloading and harvesting of high-res game sprites (PMDCollab, Hugging Face, Spriters Resource).
- train: Train the VQ-GAN reconstruction model.
- eval: Evaluate a trained model checkpoint against real test images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from spriteforge.core.alpha import threshold_alpha, despeckle_alpha
from spriteforge.core.degrade import degrade, DegradeRanges
from spriteforge.core.palette import (
    extract_palette_kmeans,
    extract_palette_median_cut,
    list_builtin_palettes,
    load_builtin_palette,
    load_palette,
    nearest_neighbor_snap,
    ordered_dither_snap,
)
from spriteforge.core.resize import area_average_downscale, resize_to_target


from spriteforge.core.io import load_image_float32, save_image_float32


def cmd_convert(args: argparse.Namespace) -> int:
    """Execute Stage A deterministic conversion pipeline."""
    print(f"[*] Loading input image: {args.input}")
    img = load_image_float32(args.input)

    # 1. Deterministic downscaling
    print(f"[*] Downscaling to target size {args.size}x{args.size} (method: {args.resize_method})...")
    if args.pre_downscale > 1.0:
        img = area_average_downscale(img, factor=args.pre_downscale)
    img_resized = resize_to_target(img, target_size=args.size, method=args.resize_method)

    # 2. Palette extraction or loading
    if args.palette_mode == "fixed":
        if not args.palette_file:
            print("[!] Error: --palette-file required when --palette-mode is 'fixed'", file=sys.stderr)
            return 1
        print(f"[*] Loading fixed palette from {args.palette_file}...")
        palette_rgb = load_palette(args.palette_file)
    elif args.palette_mode == "preset":
        print(f"[*] Loading preset palette '{args.palette_preset}'...")
        palette_rgb = load_builtin_palette(args.palette_preset)
    elif args.palette_mode == "kmeans":
        print(f"[*] Extracting {args.colors}-color palette via OKLab K-Means...")
        palette_rgb = extract_palette_kmeans(img_resized, k=args.colors, alpha_threshold=args.alpha_thresh)
    elif args.palette_mode == "median-cut":
        print(f"[*] Extracting {args.colors}-color palette via OKLab Median-Cut...")
        palette_rgb = extract_palette_median_cut(img_resized, k=args.colors, alpha_threshold=args.alpha_thresh)
    else:
        print(f"[!] Error: unknown palette mode {args.palette_mode}", file=sys.stderr)
        return 1

    # 3. Palette snapping in OKLab space
    print("[*] Snapping colors in perceptual OKLab space...")
    if args.dither:
        img_snapped = ordered_dither_snap(img_resized, palette_rgb, strength=args.dither_strength, alpha_threshold=args.alpha_thresh)
    else:
        img_snapped = nearest_neighbor_snap(img_resized, palette_rgb, alpha_threshold=args.alpha_thresh)

    # 4. Alpha thresholding & despeckling
    print("[*] Enforcing hard transparency matte...")
    img_alpha = threshold_alpha(img_snapped, threshold=args.alpha_thresh)
    if args.despeckle:
        print("[*] Despeckling isolated stray pixels...")
        img_alpha = despeckle_alpha(img_alpha, min_area=args.despeckle_min_area)

    print(f"[*] Saving output sprite to: {args.output}")
    save_image_float32(img_alpha, args.output)
    print("[+] Conversion complete!")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Generate side-by-side calibration grids for human inspection (Phase 3 gate)."""
    print(f"[*] Generating calibration grid from: {args.input}")
    img = load_image_float32(args.input)

    # Left side: deterministic downscaled real input
    real_downscaled = resize_to_target(img, target_size=args.size, method="area")
    
    # Right side: generate multiple degraded variations
    rng = np.random.default_rng(seed=args.seed)
    samples = [real_downscaled]
    
    mode = getattr(args, "mode", "all")
    if mode == "all":
        # 2 realistic low-noise, 2 color-shift only, 2 standard
        for _ in range(2):
            samples.append(degrade(real_downscaled, rng=rng, ranges=DegradeRanges.realistic_low_noise()))
        for _ in range(2):
            samples.append(degrade(real_downscaled, rng=rng, ranges=DegradeRanges.color_shift_only()))
        for _ in range(2):
            samples.append(degrade(real_downscaled, rng=rng, ranges=DegradeRanges.standard()))
    else:
        if mode == "realistic":
            ranges = DegradeRanges.realistic_low_noise()
        elif mode == "color_shift":
            ranges = DegradeRanges.color_shift_only()
        else:
            ranges = DegradeRanges.standard()
        for _ in range(args.num_samples - 1):
            samples.append(degrade(real_downscaled, rng=rng, ranges=ranges))

    # Concatenate horizontally into a review grid
    grid = np.concatenate(samples, axis=1)
    
    print(f"[*] Saving calibration review grid to: {args.output}")
    save_image_float32(grid, args.output)
    print("[+] Calibration grid saved! Review visually against real downscaled inputs.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest sprite sheets into normalized training datasets."""
    from spriteforge.data.ingest import ingest_sprite_pack
    print(f"[*] Ingesting pack from {args.input_dir} into {args.output_dir} (tier: {args.license_tier})...")
    count = ingest_sprite_pack(args.input_dir, args.output_dir, target_size=args.size, source_name=args.source)
    print(f"[+] Successfully ingested {count} sprites.")
    return 0


def cmd_scrape(args: argparse.Namespace) -> int:
    """Automated scraping and harvesting of popular game sprites."""
    from spriteforge.data.scrapers import (
        download_pmdcollab,
        download_huggingface_dataset,
        process_spriters_resource_dir,
        download_lpc_humanoids,
        download_fire_emblem_repo,
        process_ragnarok_maplestory_dir,
        download_pokeapi_repo,
        download_retro_rpg_sprites,
    )

    if args.source == "pmdcollab":
        print(f"[*] Starting automated download & slicing of PMDCollab to {args.output_dir}...")
        download_pmdcollab(args.output_dir, target_size=args.size, max_sprites=args.max_sprites, filter_keywords=args.filter, exclude_keywords=args.exclude)
    elif args.source == "huggingface":
        if not args.hf_dataset:
            print("[!] Error: --hf-dataset required when source is 'huggingface'", file=sys.stderr)
            return 1
        print(f"[*] Downloading Hugging Face dataset {args.hf_dataset} to {args.output_dir}...")
        download_huggingface_dataset(args.hf_dataset, args.output_dir, target_size=args.size, max_sprites=args.max_sprites, filter_keywords=args.filter, exclude_keywords=args.exclude)
    elif args.source == "spriters-resource":
        if not args.input_dir:
            print("[!] Error: --input-dir required when source is 'spriters-resource'", file=sys.stderr)
            return 1
        print(f"[*] Slicing raw Spriters Resource sheets from {args.input_dir} into {args.output_dir}...")
        process_spriters_resource_dir(args.input_dir, args.output_dir, target_size=args.size, max_sprites=args.max_sprites, filter_keywords=args.filter, exclude_keywords=args.exclude)
    elif args.source == "lpc":
        print(f"[*] Downloading & harvesting Universal LPC modular humanoids to {args.output_dir}...")
        download_lpc_humanoids(args.output_dir, target_size=args.size, max_sprites=args.max_sprites, filter_keywords=args.filter, exclude_keywords=args.exclude)
    elif args.source == "fire-emblem":
        print(f"[*] Downloading & harvesting Fire Emblem GBA RPG sprites to {args.output_dir}...")
        download_fire_emblem_repo(args.output_dir, target_size=args.size, max_sprites=args.max_sprites, filter_keywords=args.filter, exclude_keywords=args.exclude)
    elif args.source == "ragnarok-maplestory":
        if not args.input_dir:
            print("[!] Error: --input-dir required when source is 'ragnarok-maplestory'", file=sys.stderr)
            return 1
        print(f"[*] Harvesting Ragnarok/MapleStory chibi humanoids from {args.input_dir} into {args.output_dir}...")
        process_ragnarok_maplestory_dir(args.input_dir, args.output_dir, target_size=args.size, max_sprites=args.max_sprites, filter_keywords=args.filter, exclude_keywords=args.exclude)
    elif args.source == "pokeapi-icons":
        print(f"[*] Harvesting canonical item icons and UI objects from PokeAPI to {args.output_dir}...")
        # Clear default exclude list so item/icon keywords are not skipped
        exc = None if args.exclude == ["ui", "font", "icon", "text", "symbol", "logo", "title", "menu", "item", "fx", "effect"] else args.exclude
        download_pokeapi_repo(args.output_dir, mode="icons", target_size=args.size, max_sprites=args.max_sprites, filter_keywords=args.filter, exclude_keywords=exc)
    elif args.source == "pokeapi-overworld":
        print(f"[*] Harvesting canonical RPG overworld walking sprites from PokeAPI to {args.output_dir}...")
        download_pokeapi_repo(args.output_dir, mode="overworld", target_size=args.size, max_sprites=args.max_sprites, filter_keywords=args.filter, exclude_keywords=args.exclude)
    elif args.source == "retro-rpg":
        print(f"[*] Harvesting SNES/16-bit retro RPG character sprites to {args.output_dir}...")
        download_retro_rpg_sprites(args.output_dir, target_size=args.size, max_sprites=args.max_sprites, filter_keywords=args.filter, exclude_keywords=args.exclude)
    else:
        print(f"[!] Unknown source: {args.source}", file=sys.stderr)
        return 1
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the VQ-GAN reconstruction model."""
    from torch.utils.data import DataLoader
    from spriteforge.train.train import SpriteDataset, train_model

    print(f"[*] Initializing dataset from {args.data_dir} for size {args.size}x{args.size}...")
    degrade_profile = getattr(args, "degrade_profile", "legacy")
    print(f"[*] Degradation profile: {degrade_profile}")
    dataset = SpriteDataset(args.data_dir, target_size=args.size, apply_degradation=True, degrade_profile=degrade_profile)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    train_model(
        config_name=str(args.size),
        train_loader=loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        disc_start_epoch=args.disc_start,
        adv_weight=args.adv_weight,
        fm_weight=args.fm_weight,
        adv_ramp_epochs=args.adv_ramp_epochs,
        resume_path=args.resume
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate trained model checkpoint."""
    from torch.utils.data import DataLoader
    from spriteforge.train.train import SpriteDataset
    from spriteforge.train.evaluate import evaluate_checkpoint

    print(f"[*] Evaluating checkpoint {args.checkpoint} on dataset {args.data_dir}...")
    dataset = SpriteDataset(args.data_dir, target_size=args.size, apply_degradation=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate_checkpoint(
        args.checkpoint,
        loader,
        output_grid_path=args.output,
        num_grid_samples=args.num_samples,
        sample_mode=args.sample_mode,
        seed=args.seed,
        max_eval_samples=args.max_eval_samples
    )
    print(f"[+] Evaluation results: {metrics}")
    return 0


def cmd_slider(args: argparse.Namespace) -> int:
    """Export before/after slider PNGs from a checkpoint (Phase 5.2 content assets)."""
    from spriteforge.train.artifacts import export_before_after_sliders
    print(f"[*] Exporting before/after sliders from {args.checkpoint}...")
    written = export_before_after_sliders(
        args.checkpoint, args.input, args.output_dir, device=args.device, upscale=args.upscale
    )
    print(f"[+] Wrote {len(written)} slider assets to {args.output_dir}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    """Launch PySide6 Spriteforge Studio GUI (Phase 6)."""
    from spriteforge.app.gui import run_app
    print("[*] Launching Spriteforge Studio GUI...")
    run_app()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Build and execute the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="spriteforge",
        description="Spriteforge: AI-powered retro game sprite generation and restoration studio."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- convert ---
    p_conv = subparsers.add_parser("convert", help="Convert image to retro sprite (Phase 1 Stage A)")
    p_conv.add_argument("-i", "--input", required=True, help="Input image path")
    p_conv.add_argument("-o", "--output", required=True, help="Output sprite PNG path")
    p_conv.add_argument("-s", "--size", type=int, choices=[16, 32, 48], default=32, help="Target sprite size")
    p_conv.add_argument("--palette-mode", choices=["kmeans", "median-cut", "preset", "fixed"], default="kmeans", help="Palette extraction mode")
    p_conv.add_argument("--palette-preset", choices=list_builtin_palettes(), default="pico8", help="Bundled palette name, used when --palette-mode=preset")
    p_conv.add_argument("--palette-file", help="Path to a palette file (.json/.hex/.pal), used when --palette-mode=fixed")
    p_conv.add_argument("-c", "--colors", type=int, default=16, help="Number of colors")
    p_conv.add_argument("--dither", action="store_true", help="Enable Bayer ordered dithering")
    p_conv.add_argument("--dither-strength", type=float, default=0.05, help="Dithering strength in OKLab space")
    p_conv.add_argument("--resize-method", choices=["area", "nearest", "cubic"], default="area", help="Resize interpolation method")
    p_conv.add_argument("--pre-downscale", type=float, default=1.0, help="Area average pre-downscale factor")
    p_conv.add_argument("--alpha-thresh", type=float, default=0.5, help="Alpha threshold cutoff [0, 1]")
    p_conv.add_argument("--despeckle", action="store_true", help="Remove isolated 1-pixel stray speckles")
    p_conv.add_argument("--despeckle-min-area", type=int, default=2, help="Minimum pixel area for despeckling")
    p_conv.set_defaults(func=cmd_convert)

    # --- calibrate ---
    p_cal = subparsers.add_parser("calibrate", help="Generate degradation calibration grids (Phase 3 gate)")
    p_cal.add_argument("-i", "--input", required=True, help="Input image or clean sprite path")
    p_cal.add_argument("-o", "--output", required=True, help="Output grid PNG path")
    p_cal.add_argument("-s", "--size", type=int, choices=[16, 32, 48], default=32, help="Target sprite size")
    p_cal.add_argument("-n", "--num-samples", type=int, default=7, help="Number of grid columns to generate")
    p_cal.add_argument("--mode", choices=["all", "standard", "realistic", "color_shift"], default="all", help="Degradation preset mode")
    p_cal.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p_cal.set_defaults(func=cmd_calibrate)

    # --- ingest ---
    p_ing = subparsers.add_parser("ingest", help="Ingest and slice sprite sheets (Phase 2)")
    p_ing.add_argument("-i", "--input-dir", required=True, help="Raw asset directory")
    p_ing.add_argument("-o", "--output-dir", required=True, help="Processed dataset output directory")
    p_ing.add_argument("-s", "--size", type=int, choices=[16, 32, 48], default=32, help="Target sprite size")
    p_ing.add_argument("--source", required=True, help="Source name for provenance logging")
    p_ing.add_argument("--license-tier", choices=["cc0_public", "private_research", "custom_user"], default="cc0_public")
    p_ing.set_defaults(func=cmd_ingest)

    # --- scrape ---
    p_scr = subparsers.add_parser("scrape", help="Automated harvesting of popular game sprites for private training")
    p_scr.add_argument("--source", choices=["pmdcollab", "huggingface", "spriters-resource", "lpc", "fire-emblem", "ragnarok-maplestory", "pokeapi-icons", "pokeapi-overworld", "retro-rpg"], required=True, help="Target sprite source")
    p_scr.add_argument("-o", "--output-dir", required=True, help="Directory to save harvested sprites")
    p_scr.add_argument("-s", "--size", type=int, choices=[16, 32, 48], default=32, help="Target sprite size")
    p_scr.add_argument("--hf-dataset", help="Hugging Face dataset repo (e.g. Limbicnation/pixel-art-lora)")
    p_scr.add_argument("-i", "--input-dir", help="Local raw directory of sheets (for spriters-resource mode)")
    p_scr.add_argument("--max-sprites", type=int, help="Maximum number of sprites to harvest")
    p_scr.add_argument("--filter", nargs="+", help="Only harvest files whose path contains these keywords (e.g. Battle,Map,Walk)")
    p_scr.add_argument("--exclude", nargs="+", default=["ui", "font", "icon", "text", "symbol", "logo", "title", "menu", "item", "fx", "effect"], help="Skip files containing these keywords")
    p_scr.set_defaults(func=cmd_scrape)

    # --- train ---
    p_tr = subparsers.add_parser("train", help="Train VQ-GAN model (Phase 5)")
    p_tr.add_argument("--config", default="size_32", help="Model config name")
    p_tr.add_argument("--data-dir", required=True, help="Processed training dataset directory")
    p_tr.add_argument("-s", "--size", type=int, choices=[16, 32, 48], default=32, help="Target sprite size")
    p_tr.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p_tr.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p_tr.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p_tr.add_argument("--device", default="auto", help="Computing device (auto, cuda, mps, cpu)")
    p_tr.add_argument("-o", "--output-dir", default="checkpoints", help="Directory to save checkpoints")
    p_tr.add_argument("--save-interval", type=int, default=10, help="Epoch interval for saving checkpoints")
    p_tr.add_argument("--sample-interval", type=int, default=25, help="Epoch interval for devlog sample grids")
    p_tr.add_argument("--disc-start", type=int, default=3, help="Epoch to start GAN discriminator warmup")
    p_tr.add_argument("--adv-weight", type=float, default=1.0, help="Target adversarial loss weight (reached after ramp)")
    p_tr.add_argument("--fm-weight", type=float, default=0.2, help="Target feature-matching loss weight (reached after ramp)")
    p_tr.add_argument("--adv-ramp-epochs", type=int, default=10, help="Epochs to linearly ramp adv/fm weight from 0 to target after disc-start")
    p_tr.add_argument("--resume", help="Path to checkpoint to resume weights from")
    p_tr.add_argument("--degrade-profile", choices=["legacy", "ai_style", "ai_style_crisp"], default="legacy",
                      help="Degradation profile: 'legacy' (standard/realistic/color_shift mix), "
                           "'ai_style' (spans crisp->mush; matches real AI inputs), or "
                           "'ai_style_crisp' (crisp-heavy near-sprite specialist)")
    p_tr.set_defaults(func=cmd_train)

    # --- eval ---
    p_ev = subparsers.add_parser("eval", help="Evaluate trained checkpoint (Phase 5)")
    p_ev.add_argument("--checkpoint", required=True, help="Trained model .pt checkpoint path")
    p_ev.add_argument("-i", "--input", help="Optional test input image")
    p_ev.add_argument("-o", "--output", required=True, help="Output evaluation grid")
    p_ev.add_argument("-s", "--size", type=int, choices=[16, 32, 48], default=32, help="Target sprite size")
    p_ev.add_argument("--data-dir", default="./data_private/train_32", help="Dataset for evaluation metrics")
    p_ev.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p_ev.add_argument("--num-samples", type=int, default=24, help="Number of sprites to display in evaluation grid")
    p_ev.add_argument("--sample-mode", choices=["diverse", "random", "first"], default="diverse", help="How to sample sprites for the grid")
    p_ev.add_argument("--seed", type=int, default=42, help="Random seed for grid sampling")
    p_ev.add_argument("--max-eval-samples", type=int, default=None, help="Limit number of dataset samples for fast metric calculation")
    p_ev.set_defaults(func=cmd_eval)

    # --- slider ---
    p_sl = subparsers.add_parser("slider", help="Export before/after slider assets (Phase 5.2)")
    p_sl.add_argument("--checkpoint", required=True, help="Trained model .pt checkpoint path")
    p_sl.add_argument("-i", "--input", nargs="+", required=True, help="Input image path(s)")
    p_sl.add_argument("-o", "--output-dir", default="devlog/sliders", help="Output directory")
    p_sl.add_argument("--upscale", type=int, default=8, help="Nearest-neighbor upscale factor")
    p_sl.add_argument("--device", default="cpu", help="Computing device")
    p_sl.set_defaults(func=cmd_slider)

    # --- gui ---
    p_gui = subparsers.add_parser("gui", help="Launch PySide6 Spriteforge Studio GUI (Phase 6)")
    p_gui.set_defaults(func=cmd_gui)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
