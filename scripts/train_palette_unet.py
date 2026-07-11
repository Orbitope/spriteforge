"""CLI entry point for E1 palette-index classification training. See
spriteforge/train/train_palette.py for the design notes and open questions
(palette-conditioning source at inference time is not yet resolved)."""

from __future__ import annotations

import argparse

from spriteforge.train.train_palette import train_palette_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the E1 palette-index classification model")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-colors", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--cache-dir", default=None, help="Pre-computed palette cache dir (auto-derived if omitted)")
    args = parser.parse_args()

    train_palette_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_colors=args.num_colors,
        hidden_channels=args.hidden_channels,
        device=args.device,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
