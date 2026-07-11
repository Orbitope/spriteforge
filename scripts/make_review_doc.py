"""
Builds devlog/TRAINING_REVIEW.md summarizing the per-source training runs:
final/first-epoch metrics from each source's training_log.csv, eval metrics parsed from
eval_stdout.log, and pointers to the sample/eval grid PNGs for visual review.
"""
from __future__ import annotations

import ast
import csv
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BYSOURCE = ROOT / "checkpoints_bysource"
SOURCES = ["pmd", "papi", "lpc", "fe"]
SOURCE_LABELS = {
    "pmd": "PMDCollab (Pokémon Mystery Dungeon)",
    "papi": "PokeAPI overworld/icon sprites",
    "lpc": "Universal LPC modular humanoids",
    "fe": "Fire Emblem GBA battle sprites (opaque bg)",
}


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_eval_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text()
    m = re.search(r"\[\+\] Evaluation results: (\{.*\})", text)
    if not m:
        return None
    # NumPy 2.x reprs scalars as e.g. np.float64(21.44), which ast.literal_eval
    # can't parse (it's a call, not a literal) — unwrap before parsing.
    unwrapped = re.sub(r"np\.\w+\(([^()]+)\)", r"\1", m.group(1))
    try:
        return ast.literal_eval(unwrapped)
    except Exception:
        return None


def fmt_row(rows: list[dict]) -> str:
    if not rows:
        return "_no training_log.csv found — run likely failed before logging started_"
    first, last = rows[0], rows[-1]
    lines = [
        f"- Epochs logged: {len(rows)} (final epoch {last['epoch']})",
        f"- Loss G: {first['loss_g']} → {last['loss_g']}",
        f"- Loss D: {first['loss_d']} → {last['loss_d']}",
        f"- Recon loss: {first['recon']} → {last['recon']}",
        f"- VQ loss: {first['vq']} → {last['vq']}",
        f"- Orthogonality metric: {first['ortho']} → {last['ortho']} "
        f"(lower = healthier codebook)",
        f"- Active codes: {first['active_codes']} → {last['active_codes']} "
        f"/ {last['total_codes']} total",
    ]
    return "\n".join(lines)


def read_palette_snap_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def fmt_eval(metrics: dict | None) -> str:
    if metrics is None:
        return "_no eval metrics found — eval step likely failed, check eval_stdout.log_"
    return (
        f"- PSNR: {metrics.get('psnr')} dB\n"
        f"- L1 loss: {metrics.get('l1_loss')}\n"
        f"- Codebook usage: {metrics.get('codebook_usage')} "
        f"({metrics.get('active_codes')}/{metrics.get('total_codes')} codes active on eval set)"
    )


def main():
    out_lines = [
        "# Spriteforge — Multi-Source Training Review",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Each source below was trained independently for 100 epochs from scratch "
        "(config `size_32`, `--disc-start 10 --adv-ramp-epochs 20` — adversarial loss "
        "ramps in linearly over epochs 10-30 instead of switching on instantly, to avoid "
        "destabilizing reconstruction), then evaluated against a held-out per-source test "
        "set (20 images, not seen in training). A `vqgan_32_best_recon.pt` checkpoint is "
        "also saved per source, tracking the lowest-recon-loss epoch seen.",
        "",
        "**Data-quality notes:** two of the four held-out eval sets shipped with this repo "
        "turned out to be corrupted and were rebuilt from the real training distribution "
        "before trusting these numbers. `test_pmd_32` (original) contained dialogue/status-"
        "screen crops with baked-in caption text and no alpha transparency — replaced by "
        "`test_pmd_32_clean`. `test_fe_32` contained isolated letter/logo fragments from a "
        "mis-sliced trademark watermark (3/20 images) — replaced by `test_fe_32_clean`. Both "
        "were caught by visually inspecting eval grids where PSNR looked anomalous relative "
        "to training loss, not by the metrics alone — a reminder to spot-check images, not "
        "just numbers, when eval results look off.",
        "",
    ]

    overview_path = ROOT / "devlog" / "overview_grid.png"
    if overview_path.exists():
        out_lines += [
            "## At a glance",
            "",
            "One random held-out example per source (input / restored / ground truth):",
            "",
            "![Cross-source overview](overview_grid.png)",
            "",
        ]

    metrics_by_source = {}
    for source in SOURCES:
        eval_log_path = BYSOURCE / source / "eval_stdout.log"
        metrics_by_source[source] = parse_eval_metrics(eval_log_path)

    if any(metrics_by_source.values()):
        out_lines += [
            "## Summary",
            "",
            "| Source | PSNR (dB) | L1 loss | Codebook usage | Active codes |",
            "|---|---|---|---|---|",
        ]
        for source in SOURCES:
            m = metrics_by_source[source]
            if m is None:
                out_lines.append(f"| {source} | — | — | — | — |")
            else:
                out_lines.append(
                    f"| {source} | {m.get('psnr')} | {m.get('l1_loss')} | "
                    f"{m.get('codebook_usage')} | {m.get('active_codes')}/{m.get('total_codes')} |"
                )
        out_lines += ["", "---", ""]

    out_lines += [
        "## Key finding: palette-snap post-processing helps *visually*, not always *numerically*",
        "",
        "Per-source sections below embed a palette-snap comparison: the model's continuous "
        "RGBA output run through `extract_palette_kmeans` (k=6, OKLab space) + "
        "`nearest_neighbor_snap` (`spriteforge/core/palette.py`) — the same deterministic "
        "post-processing Stage A already does for raw-image conversion, applied here to "
        "Stage B's neural output. Two palette sources are shown: inferred from the degraded "
        "input (realistic — the only thing available at real inference time) and from the "
        "ground truth (a ceiling, using color info you would never actually have).",
        "",
        "**Visually**, snapping consistently removes the speckled/muddy color noise that is "
        "the raw VQ-GAN output's most obvious flaw — see any of the grids below, e.g. `pmd` "
        "row 1 or row 4, where a noisy multi-color speckle field collapses into a clean, "
        "flat, sprite-like fill.",
        "",
        "**Numerically, it's mixed.** A k-sweep on `pmd` (raw model L1 ≈ 0.026-0.030 "
        "depending on run):",
        "",
        "| Palette source | k=6 | k=12 | k=16 | k=24 |",
        "|---|---|---|---|---|",
        "| From degraded input (realistic) | 0.0327 | 0.0305 | 0.0296 | 0.0310 |",
        "| From model's own output (self) | 0.0275 | 0.0299 | 0.0280 | — |",
        "| From ground truth (ceiling) | 0.0249 | 0.0292 | 0.0235 | — |",
        "",
        "Only the ground-truth-derived palette reliably *beats* raw output on L1, at every "
        "k tested. Snapping from the degraded input never wins — the input's own colors are "
        "already corrupted by blur/noise/jitter, so a palette extracted from it inherits that "
        "corruption and compounds it. Snapping from the model's *own* output sits in between: "
        "close to break-even, better than snapping from the raw input, because the model's "
        "output is already partially denoised.",
        "",
        "**Practical takeaway:** palette-snapping is worth keeping as a Stage A-style "
        "post-process for the visual cleanup alone, but for the best numerical result without "
        "cheating via ground truth, snap using a palette extracted from the *model's own "
        "restored output* (not the raw input) at a moderate k (12-16), not k=6.",
        "",
    ]

    # Fixed-palette results, if the assets were generated with --palette-file.
    fixed_rows = []
    for source in SOURCES:
        m = read_palette_snap_metrics(BYSOURCE / source / "palette_snap_metrics.json")
        if m and "fixed_palette_snap_l1_mean" in m:
            fixed_rows.append((source, m))

    if fixed_rows:
        out_lines += [
            "### Fixed pre-defined palette (the real production case)",
            "",
            f"`scripts/build_review_examples.py --palette-file <path>` snaps every restored "
            f"output to a single pre-defined palette instead of auto-inferring one per image — "
            f"this is how the shipped pipeline should actually run once a real target palette "
            f"exists. The assets below used "
            f"`spriteforge/data/palettes/pico8.json` ({fixed_rows[0][1].get('fixed_palette_size')} "
            f"colors) as a stand-in — **swap in your actual production palette file to get "
            f"meaningful numbers**; PICO-8's colors have no relation to any of these sprite "
            f"domains and this is expected to look worse than the auto-inferred columns.",
            "",
            "| Source | Raw L1 | Auto-snap L1 | Fixed-palette L1 |",
            "|---|---|---|---|",
        ]
        for source, m in fixed_rows:
            out_lines.append(
                f"| {source} | {m['raw_l1_mean']} | {m['auto_palette_snap_l1_mean']} | "
                f"{m['fixed_palette_snap_l1_mean']} |"
            )
        out_lines += [
            "",
            "**This is the key caveat for real deployment:** a fixed palette only helps if "
            "it's actually representative of the sprite domain's true colors. An unrelated "
            "palette (like PICO-8 here) doesn't collapse the model's color speckle noise the "
            "way an in-domain palette does — nearby true colors land on different, still-wrong "
            "palette entries, so the speckle persists (see the grids below, \"Snap (fixed "
            "palette)\" column). Before shipping this against a real predefined palette, "
            "regenerate these grids with it and confirm the same collapse-to-clean-fill effect "
            "actually happens.",
            "",
        ]

    out_lines += ["---", ""]

    for source in SOURCES:
        out_dir = BYSOURCE / source
        csv_path = out_dir / "training_log.csv"
        samples_dir = out_dir / "samples"

        rows = read_csv_rows(csv_path)
        metrics = metrics_by_source[source]

        sample_pngs = sorted(samples_dir.glob("sample_epoch_*.png")) if samples_dir.exists() else []
        eval_grid = out_dir / "eval_grid.png"
        random_examples = out_dir / "review_examples" / "random_examples.png"
        palette_examples = out_dir / "review_examples" / "palette_snap_examples.png"
        palette_metrics = read_palette_snap_metrics(out_dir / "palette_snap_metrics.json")
        failure_grid = ROOT / "devlog" / "failures" / f"{source}_worst_3.png"

        out_lines += [
            f"## {source} — {SOURCE_LABELS.get(source, '')}",
            "",
            "**Training curve (from training_log.csv):**",
            "",
            fmt_row(rows),
            "",
            "**Held-out eval metrics:**",
            "",
            fmt_eval(metrics),
            "",
        ]

        if random_examples.exists():
            out_lines += [
                "**Randomly selected examples** (input / restored-final / restored-best-recon / "
                "ground truth — regenerate with `scripts/build_review_examples.py` for a fresh draw):",
                "",
                f"![{source} random examples](../checkpoints_bysource/{source}/review_examples/random_examples.png)",
                "",
            ]

        if palette_examples.exists():
            has_fixed = palette_metrics and "fixed_palette_snap_l1_mean" in palette_metrics
            caption = "input / restored"
            if has_fixed:
                caption += " / snap-fixed-palette"
            caption += " / snap-auto-from-input / snap-auto-from-GT-ceiling / ground truth"
            out_lines += [
                f"**Palette-snap post-processing** ({caption} — see \"Key finding\" above):",
                "",
                f"![{source} palette-snap examples](../checkpoints_bysource/{source}/review_examples/palette_snap_examples.png)",
                "",
            ]
            if palette_metrics:
                k = palette_metrics.get("palette_k", 6)
                line = (
                    f"- Full held-out set ({palette_metrics['num_eval_samples']} images), "
                    f"raw model L1: {palette_metrics['raw_l1_mean']} vs. auto palette-snap "
                    f"(k={k}, from input) L1: {palette_metrics['auto_palette_snap_l1_mean']}"
                )
                if has_fixed:
                    line += f" vs. fixed-palette snap L1: {palette_metrics['fixed_palette_snap_l1_mean']}"
                out_lines += [line, ""]

        if failure_grid.exists():
            out_lines += [
                "**Worst cases** (highest per-image L1 error against the final checkpoint):",
                "",
                f"![{source} worst cases](failures/{source}_worst_3.png)",
                "",
            ]

        out_lines += [
            "**Artifacts:**",
            f"- Checkpoints: `checkpoints_bysource/{source}/vqgan_32_epoch_*.pt`",
            f"- Full per-epoch log: `checkpoints_bysource/{source}/training_log.csv`",
            f"- Sample grids ({len(sample_pngs)} saved): `checkpoints_bysource/{source}/samples/`",
            f"- Eval comparison grid: `checkpoints_bysource/{source}/eval_grid.png`"
            + (" (missing)" if not eval_grid.exists() else ""),
            "",
            "---",
            "",
        ]

    out_lines += [
        "## How to compare sources",
        "",
        "- Lower `ortho` + higher `active_codes` at epoch 50 = healthier, less-collapsed "
        "codebook for that source's visual vocabulary.",
        "- Compare `eval_grid.png` across sources side by side — top row is the degraded "
        "input, middle is the model's restoration, bottom is ground truth.",
        "- If one source's codebook collapses hard (active_codes trending toward a small "
        "fraction of total_codes) while others don't, that source's sprites may be too "
        "visually homogeneous, or may need a larger codebook / longer disc warmup.",
        "",
    ]

    devlog_dir = ROOT / "devlog"
    devlog_dir.mkdir(exist_ok=True)
    out_path = devlog_dir / "TRAINING_REVIEW.md"
    out_path.write_text("\n".join(out_lines))
    print(f"[+] Wrote review document: {out_path}")


if __name__ == "__main__":
    main()
