# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Degradation core — turns a clean sprite into a plausible "downscaled real image"
approximation, so a reconstruction model learns to invert exactly these artifacts.

Design contract:
    - Pure functions. No file I/O, no UI, no global state.
    - Input/output are float32 RGBA arrays in [0, 1], shape (H, W, 4).
    - All randomness flows through an explicit np.random.Generator (reproducible).
    - Each primitive targets ONE real-input artifact. Presets/pipelines compose them.

This is the phase-1 crux: get these artifacts right and the model bridges
"trained on sprites only" -> "works on real downscaled inputs".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class DegradeRanges:
    """Range-based knobs. Each sample draws uniformly (or per-primitive) from these.

    Ranges are chosen to bracket what a real photo/render looks like once it has
    been resized down toward the sprite grid: soft edges, slight color drift,
    anti-aliasing, mild noise/compression, sub-pixel misalignment.
    """
    # Probability each primitive fires on a given sample (independent gates).
    p_blur: float = 0.85
    p_color_jitter: float = 0.9
    p_antialias_edges: float = 0.8
    p_noise: float = 0.3
    p_jpeg: float = 0.3
    p_palette_bleed: float = 0.5
    p_subpixel_shift: float = 0.6
    p_alpha_soften: float = 0.5
    p_inject_background: float = 0.5

    # --- v2 coverage additions ---
    # AI-source artifacts (local color inconsistency is the signature one).
    # Re-enabled at low probability (Stage 2, methodology review Part 1c): these
    # were the one artifact family motivating the project and were at p=0.0 in
    # every preset. Kept low so hues stay mostly faithful on the 80%+ of samples
    # where they don't fire.
    p_local_color_drift: float = 0.15
    p_region_texture: float = 0.2
    p_palette_inflation: float = 0.15
    # Both-path additions.
    p_posterize: float = 0.4
    p_alpha_morph: float = 0.5
    p_motion_blur: float = 0.2

    # Blur: gaussian sigma in *target-pixel* units (small; these are tiny images).
    blur_sigma: tuple[float, float] = (0.3, 0.9)

    # Color jitter: disabled hue and saturation shifts to keep colors 100% faithful!
    brightness: tuple[float, float] = (-0.08, 0.08)
    contrast: tuple[float, float] = (0.85, 1.15)
    hue_deg: tuple[float, float] = (0.0, 0.0)
    saturation: tuple[float, float] = (1.0, 1.0)

    # Edge anti-aliasing: fraction of a pixel the AA band spans.
    aa_strength: tuple[float, float] = (0.2, 0.8)

    # Noise (gaussian, std in [0,1] color units) — significantly reduced for cleaner training!
    noise_std: tuple[float, float] = (0.002, 0.012)

    # JPEG quality (lower = more blocking/ringing) — raised minimum to avoid harsh ringing!
    jpeg_quality: tuple[int, int] = (75, 95)

    # Palette bleed: neighbor-color mixing amount (adjacent flat regions bleed).
    bleed_amount: tuple[float, float] = (0.08, 0.25)

    # Sub-pixel shift, in target-pixel units (misalignment to the sprite grid).
    shift: tuple[float, float] = (-0.5, 0.5)

    # Alpha edge softening (matte halo), sigma in target-pixel units.
    alpha_sigma: tuple[float, float] = (0.2, 0.7)

    # --- v2 coverage additions ---
    # Local color drift: smooth low-freq hue/brightness variation WITHIN flat
    # regions. The signature AI-source artifact. Amplitude in [0,1] color units;
    # scale = spatial smoothness (larger = broader, gentler patches).
    drift_amount: tuple[float, float] = (0.03, 0.10)
    drift_scale: tuple[float, float] = (2.0, 6.0)

    # Region texture: faint high-freq detail where a sprite would be flat fill.
    texture_amount: tuple[float, float] = (0.01, 0.04)

    # Palette inflation: jitter to push near-identical colors apart (AI images
    # use hundreds of colors where a sprite uses a handful). Per-pixel std.
    inflation_std: tuple[float, float] = (0.005, 0.015)

    # Posterize: bits per channel retained (fewer = harder crunch).
    posterize_bits: tuple[int, int] = (4, 6)

    # Alpha morph: dilate/erode radius in target-pixels (matte too loose/tight).
    # Sampled symmetric; negative = erode, positive = dilate.
    alpha_morph_px: tuple[int, int] = (-1, 1)

    # Motion blur: kernel length in target-pixels; angle drawn uniformly.
    motion_len: tuple[float, float] = (1.2, 2.5)

    @classmethod
    def realistic_low_noise(cls) -> DegradeRanges:
        """Slightly less noise versions that are closer to actuality (slight blur, ultra low noise, zero hue shift)."""
        return cls(
            p_blur=0.7, blur_sigma=(0.2, 0.6),
            p_color_jitter=0.85, brightness=(-0.06, 0.06), contrast=(0.90, 1.10), hue_deg=(0.0, 0.0), saturation=(1.0, 1.0),
            p_antialias_edges=0.6, aa_strength=(0.2, 0.5),
            p_noise=0.1, noise_std=(0.001, 0.006),
            p_jpeg=0.15, jpeg_quality=(85, 98),
            p_palette_bleed=0.4, bleed_amount=(0.05, 0.15),
            p_subpixel_shift=0.3, shift=(-0.2, 0.2),
            p_alpha_soften=0.3, alpha_sigma=(0.1, 0.4),
            p_inject_background=0.4,
            p_local_color_drift=0.08, drift_amount=(0.02, 0.08), drift_scale=(3.0, 6.0),
            p_region_texture=0.1, texture_amount=(0.005, 0.02),
            p_palette_inflation=0.08, inflation_std=(0.005, 0.01),
            p_posterize=0.2, posterize_bits=(5, 6),
            p_alpha_morph=0.2, alpha_morph_px=(-1, 1),
            p_motion_blur=0.1, motion_len=(1.1, 1.8)
        )

    @classmethod
    def color_shift_only(cls) -> DegradeRanges:
        """Just have contrast/brightness/palette bleed without hue/saturation distortion."""
        return cls(
            p_blur=0.0,
            p_color_jitter=1.0, brightness=(-0.10, 0.10), contrast=(0.80, 1.20), hue_deg=(0.0, 0.0), saturation=(1.0, 1.0),
            p_antialias_edges=0.0,
            p_noise=0.0,
            p_jpeg=0.0,
            p_palette_bleed=0.8, bleed_amount=(0.10, 0.35),
            p_subpixel_shift=0.0,
            p_alpha_soften=0.0,
            p_local_color_drift=0.12, drift_amount=(0.03, 0.10), drift_scale=(2.0, 5.0),
            p_region_texture=0.0,
            p_palette_inflation=0.12, inflation_std=(0.005, 0.015),
            p_posterize=0.0,
            p_alpha_morph=0.0,
            p_motion_blur=0.0
        )

    @classmethod
    def ai_style_source(cls) -> DegradeRanges:
        """Model the gap between a clean sprite and an *AI render of that sprite*.

        Motivated by the real-input eval set (data_private/real_eval_raw): those
        images are crisp, detailed, near-sprite characters sitting on a flat/gradient
        background — NOT the blurry mush the `standard()` profile produces. Feeding a
        model trained on mush a crisp input is out-of-distribution and it "restores"
        problems that aren't there (devlog: 2026-07-09 domain-gap diagnosis).

        This profile therefore:
          - Emphasizes the AI-signature artifacts (local color drift, region texture,
            palette inflation) that turn a flat sprite fill into AI-style richness.
          - Always injects a background (real AI outputs are never transparent).
          - Keeps optics light: gentle anti-alias/softness, minimal gaussian blur, no
            motion blur, no harsh JPEG — structure stays intact.
          - Leaves hue/saturation faithful (color is recovered by palette-snap).
        """
        return cls(
            p_inject_background=1.0,
            p_blur=0.25, blur_sigma=(0.2, 0.5),
            p_motion_blur=0.0,
            p_color_jitter=0.8, brightness=(-0.06, 0.06), contrast=(0.90, 1.12),
            hue_deg=(0.0, 0.0), saturation=(1.0, 1.0),
            p_antialias_edges=0.7, aa_strength=(0.2, 0.5),
            p_subpixel_shift=0.5, shift=(-0.4, 0.4),
            p_alpha_soften=0.6, alpha_sigma=(0.2, 0.6),
            p_alpha_morph=0.3, alpha_morph_px=(-1, 1),
            p_palette_bleed=0.5, bleed_amount=(0.06, 0.18),
            p_noise=0.2, noise_std=(0.002, 0.010),
            p_jpeg=0.15, jpeg_quality=(82, 96),
            # AI-signature artifacts — the whole point of this profile — turned up.
            p_local_color_drift=0.7, drift_amount=(0.04, 0.12), drift_scale=(2.0, 6.0),
            p_region_texture=0.7, texture_amount=(0.02, 0.06),
            p_palette_inflation=0.8, inflation_std=(0.008, 0.020),
            p_posterize=0.2, posterize_bits=(5, 6),
        )

    @classmethod
    def standard(cls) -> DegradeRanges:
        """Standard full randomized degradation."""
        return cls()



# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _u(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def _split(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb, alpha) views/copies as float32."""
    return img[..., :3].copy(), img[..., 3:4].copy()


def _merge(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.clip(np.concatenate([rgb, alpha], axis=-1), 0.0, 1.0).astype(np.float32)


def _rgb_to_u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _u8_to_rgb(u8: np.ndarray) -> np.ndarray:
    return (u8.astype(np.float32)) / 255.0


# --------------------------------------------------------------------------- #
# Primitives — each targets one real-input artifact
# --------------------------------------------------------------------------- #

def blur(img: np.ndarray, rng: np.random.Generator, r: DegradeRanges) -> np.ndarray:
    """Softening from downscale/optics. Blurs RGB; leaves alpha to alpha_soften."""
    sigma = _u(rng, r.blur_sigma)
    rgb, a = _split(img)
    k = max(3, int(sigma * 4) | 1)  # odd kernel
    rgb = cv2.GaussianBlur(rgb, (k, k), sigmaX=sigma, sigmaY=sigma,
                           borderType=cv2.BORDER_REPLICATE)
    return _merge(rgb, a)


def color_jitter(img: np.ndarray, rng: np.random.Generator, r: DegradeRanges) -> np.ndarray:
    """Exposure / white-balance / saturation drift of a real capture or render."""
    rgb, a = _split(img)
    # brightness + contrast around 0.5 mid-gray
    b = _u(rng, r.brightness)
    c = _u(rng, r.contrast)
    rgb = np.clip((rgb - 0.5) * c + 0.5 + b, 0.0, 1.0)
    # hue + saturation in HSV
    hsv = cv2.cvtColor(_rgb_to_u8(rgb), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + _u(rng, r.hue_deg) / 2.0) % 180.0  # OpenCV H in [0,180)
    hsv[..., 1] = np.clip(hsv[..., 1] * _u(rng, r.saturation), 0, 255)
    rgb = _u8_to_rgb(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    return _merge(rgb, a)


def antialias_edges(img: np.ndarray, rng: np.random.Generator, r: DegradeRanges) -> np.ndarray:
    """Real downscales anti-alias hard sprite edges. Blend across detected edges."""
    strength = _u(rng, r.aa_strength)
    rgb, a = _split(img)
    gray = cv2.cvtColor(_rgb_to_u8(rgb), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    edges = cv2.dilate(edges, np.ones((2, 2), np.float32))
    soft = cv2.GaussianBlur(rgb, (3, 3), sigmaX=0.8, borderType=cv2.BORDER_REPLICATE)
    m = (edges[..., None] * strength)
    rgb = rgb * (1 - m) + soft * m
    return _merge(rgb, a)


def noise(img: np.ndarray, rng: np.random.Generator, r: DegradeRanges) -> np.ndarray:
    """Sensor/quantization noise."""
    std = _u(rng, r.noise_std)
    rgb, a = _split(img)
    rgb = rgb + rng.normal(0.0, std, size=rgb.shape).astype(np.float32)
    return _merge(rgb, a)


def jpeg(img: np.ndarray, rng: np.random.Generator, r: DegradeRanges) -> np.ndarray:
    """Blocking/ringing from lossy source images. Alpha preserved."""
    q = int(rng.integers(r.jpeg_quality[0], r.jpeg_quality[1] + 1))
    rgb, a = _split(img)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(_rgb_to_u8(rgb), cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        rgb = _u8_to_rgb(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB))
    return _merge(rgb, a)


def palette_bleed(img: np.ndarray, rng: np.random.Generator, r: DegradeRanges) -> np.ndarray:
    """Adjacent flat regions bleed into each other under real resampling."""
    amt = _u(rng, r.bleed_amount)
    rgb, a = _split(img)
    bled = cv2.blur(rgb, (2, 2))  # box average = neighbor mixing
    rgb = rgb * (1 - amt) + bled * amt
    return _merge(rgb, a)


def subpixel_shift(img: np.ndarray, rng: np.random.Generator, r: DegradeRanges) -> np.ndarray:
    """Sprite grid rarely aligns to the source pixel grid; shift sub-pixel."""
    dx = _u(rng, r.shift)
    dy = _u(rng, r.shift)
    h, w = img.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return _merge(out[..., :3], out[..., 3:4])


def alpha_soften(img: np.ndarray, rng: np.random.Generator, r: DegradeRanges) -> np.ndarray:
    """Real inputs have soft/haloed alpha edges, not the sprite's hard cutout."""
    sigma = _u(rng, r.alpha_sigma)
    rgb, a = _split(img)
    k = max(3, int(sigma * 4) | 1)
    a = cv2.GaussianBlur(a, (k, k), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    if a.ndim == 2:
        a = a[..., None]
    return _merge(rgb, a)


def _lowfreq_field(shape, rng, scale):
    """Smooth low-frequency field in [-1,1], via upsampled coarse noise.

    `scale` sets the coarse-grid size relative to the image: larger scale = finer
    grid = smaller patches. We keep it deliberately coarse for broad, gentle drift.
    """
    h, w = shape
    g = max(2, int(round(scale)))
    coarse = rng.standard_normal((g, g)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    m = np.abs(field).max()
    return field / m if m > 1e-6 else field


def local_color_drift(img, rng, r):
    """Smooth hue/brightness variation WITHIN regions that should be flat.

    The signature AI-source artifact: a 'flat' fill subtly wanders in color.
    Independent low-freq field per channel, applied only where opaque.
    """
    amt = _u(rng, r.drift_amount)
    scale = _u(rng, r.drift_scale)
    rgb, a = _split(img)
    for c in range(3):
        field = _lowfreq_field(rgb.shape[:2], rng, scale)
        rgb[..., c] = rgb[..., c] + amt * field
    return _merge(rgb, a)


def region_texture(img, rng, r):
    """Faint high-freq detail where a sprite would be a solid fill."""
    amt = _u(rng, r.texture_amount)
    rgb, a = _split(img)
    tex = rng.standard_normal(rgb.shape[:2]).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (3, 3), 0.6)  # slightly correlated, not pure salt
    rgb = rgb + amt * tex[..., None]
    return _merge(rgb, a)


def palette_inflation(img, rng, r):
    """Push near-identical colors apart — mimic AI images' huge color counts.

    Per-pixel independent jitter; unlike local_color_drift this is high-freq and
    breaks exact color equality, which is what a real (non-sprite) source lacks.
    """
    std = _u(rng, r.inflation_std)
    rgb, a = _split(img)
    rgb = rgb + rng.normal(0.0, std, size=rgb.shape).astype(np.float32)
    return _merge(rgb, a)


def posterize(img, rng, r):
    """Bit-depth crunch — mimics already-quantized / banded sources."""
    bits = int(rng.integers(r.posterize_bits[0], r.posterize_bits[1] + 1))
    levels = (1 << bits) - 1
    rgb, a = _split(img)
    rgb = np.round(rgb * levels) / levels
    return _merge(rgb, a)


def alpha_morph(img, rng, r):
    """Dilate/erode the matte — cutout pulled too loose or too tight."""
    px = int(rng.integers(r.alpha_morph_px[0], r.alpha_morph_px[1] + 1))
    if px == 0:
        return img
    rgb, a = _split(img)
    k = np.ones((abs(px) * 2 + 1, abs(px) * 2 + 1), np.uint8)
    a2 = a[..., 0]
    a2 = cv2.dilate(a2, k) if px > 0 else cv2.erode(a2, k)
    return _merge(rgb, a2[..., None])


def motion_blur(img, rng, r):
    """Directional blur — subject/camera motion; beyond isotropic gaussian."""
    length = _u(rng, r.motion_len)
    angle = rng.uniform(0, np.pi)
    n = max(3, int(length) | 1)
    kern = np.zeros((n, n), np.float32)
    cx = cy = n // 2
    dx, dy = np.cos(angle), np.sin(angle)
    for t in np.linspace(-length / 2, length / 2, n * 2):
        x = int(round(cx + dx * t)); y = int(round(cy + dy * t))
        if 0 <= x < n and 0 <= y < n:
            kern[y, x] = 1.0
    s = kern.sum()
    if s < 1e-6:
        return img
    kern /= s
    out = cv2.filter2D(img, -1, kern, borderType=cv2.BORDER_REPLICATE)
    return _merge(out[..., :3], out[..., 3:4])


def inject_background(img: np.ndarray, rng: np.random.Generator, r: DegradeRanges) -> np.ndarray:
    """Simulate AI image generators placing characters on solid, noisy, or tinted backgrounds instead of true transparency.
    
    Composites the sprite over a synthetic background and sets alpha to 1.0.
    """
    if img.shape[-1] != 4:
        return img
    
    rgb, a = _split(img)
    if a.max() == 0:
        return img  # completely empty image
        
    h, w = rgb.shape[:2]
    
    # Choose background type:
    # 40% chance of light/off-white background (very common for AI sprites)
    # 30% chance of dark/black/charcoal background
    # 30% chance of colored/pastel/chroma-key background
    bg_type = rng.random()
    if bg_type < 0.4:
        base_col = rng.uniform(0.75, 1.0, size=(1, 1, 3)).astype(np.float32)
    elif bg_type < 0.7:
        base_col = rng.uniform(0.0, 0.25, size=(1, 1, 3)).astype(np.float32)
    else:
        base_col = rng.uniform(0.1, 0.9, size=(1, 1, 3)).astype(np.float32)
        
    bg = np.tile(base_col, (h, w, 1))
    
    # Optionally add slight low-frequency color drift or noise to the background
    if rng.random() < 0.5:
        noise_std = _u(rng, (0.005, 0.03))
        bg = np.clip(bg + rng.normal(0.0, noise_std, size=bg.shape).astype(np.float32), 0.0, 1.0)
        
    comp_rgb = rgb * a + bg * (1.0 - a)
    comp_a = np.ones((h, w, 1), dtype=np.float32)
    
    return _merge(comp_rgb, comp_a)


# Ordered pipeline: physically-motivated sequence (color -> optics -> sampling -> codec).
_PIPELINE = [
    # Background injection first: simulate AI generating on a non-transparent canvas!
    ("inject_background", inject_background, "p_inject_background"),
    # Color-domain (global then local) — happens "on the surface" before optics.
    ("color_jitter", color_jitter, "p_color_jitter"),
    ("local_color_drift", local_color_drift, "p_local_color_drift"),
    ("region_texture", region_texture, "p_region_texture"),
    ("palette_inflation", palette_inflation, "p_palette_inflation"),
    ("palette_bleed", palette_bleed, "p_palette_bleed"),
    ("posterize", posterize, "p_posterize"),
    # Geometry / sampling.
    ("antialias_edges", antialias_edges, "p_antialias_edges"),
    ("subpixel_shift", subpixel_shift, "p_subpixel_shift"),
    # Optics.
    ("motion_blur", motion_blur, "p_motion_blur"),
    ("blur", blur, "p_blur"),
    # Matte.
    ("alpha_morph", alpha_morph, "p_alpha_morph"),
    ("alpha_soften", alpha_soften, "p_alpha_soften"),
    # Sensor / codec (last).
    ("noise", noise, "p_noise"),
    ("jpeg", jpeg, "p_jpeg"),
]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def degrade(
    sprite: np.ndarray,
    rng: np.random.Generator | None = None,
    ranges: DegradeRanges | None = None,
    return_log: bool = False,
    multiscale: bool = False,
    multiscale_factor_range: tuple[int, int] = (4, 8),
):
    """Apply a randomized degradation chain to a clean sprite.

    Args:
        sprite: (H, W, 4) or (H, W, 3) float [0,1] or uint8. RGB assumed opaque
            if no alpha channel is present.
        rng: explicit Generator for reproducibility. Defaults to fresh entropy.
        ranges: DegradeRanges knobs. Defaults to DegradeRanges().
        return_log: if True, also return the list of primitives that fired.
        multiscale: if True, run the codec/optics primitives (motion_blur, blur, noise,
            jpeg) at an upscaled resolution and area-downscale back, instead of applying
            them directly at the sprite's native size. See
            devlog/2026-07-08-random-pool-samples.md methodology review, Part 1a: JPEG
            blocks are 8x8 — applied directly to a 32x32 sprite, one block covers a quarter
            of the image, an artifact scale that never occurs in a real pipeline where
            compression happens at source resolution and the result is downscaled after.
            Same argument for sensor noise (area-averaging during downscale reduces and
            smooths per-pixel noise magnitude — applying noise post-downscale skips that).
            **Default False** — this changes the degradation distribution, so it's opt-in
            rather than silently changing behavior for existing callers/training runs that
            depend on the original distribution for a controlled comparison.
        multiscale_factor_range: nearest-upscale factor sampled uniformly (int) from this
            range when multiscale=True.

    Returns:
        degraded (H, W, 4) float32 in [0,1], or (degraded, log) if return_log.
    """
    rng = rng or np.random.default_rng()
    ranges = ranges or DegradeRanges()

    img = _coerce(sprite)
    log: list[str] = []

    if not multiscale:
        # Original, unmodified behavior — every existing caller (training runs already in
        # progress or completed) gets byte-identical results, since this branch is untouched.
        for name, fn, pkey in _PIPELINE:
            if rng.random() < getattr(ranges, pkey):
                img = fn(img, rng, ranges)
                log.append(name)
    else:
        native_size = img.shape[0]
        for name, fn, pkey in _PIPELINE:
            if name in _SCALE_GROUP_NAMES:
                continue  # deferred to _run_scale_group below, at upscaled resolution
            if rng.random() < getattr(ranges, pkey):
                img = fn(img, rng, ranges)
                log.append(name)
            if name == _LAST_PRE_SCALE_NAME:
                img = _run_scale_group(img, rng, ranges, native_size, multiscale_factor_range, log)

    img = _merge(img[..., :3], img[..., 3:4])
    return (img, log) if return_log else img


# Primitives applied at an upscaled resolution (then area-downscaled back) when
# multiscale=True — see degrade()'s docstring. Everything else stays at native resolution.
_SCALE_GROUP_NAMES = {"motion_blur", "blur", "noise", "jpeg"}
_LAST_PRE_SCALE_NAME = "subpixel_shift"  # scale-group runs right after this point in the pipeline


def _run_scale_group(img, rng, ranges, native_size, factor_range, log):
    """Upscale (nearest), run the codec/optics primitives in their original pipeline order,
    area-downscale back to native_size."""
    factor = int(rng.integers(factor_range[0], factor_range[1] + 1))
    up_size = native_size * factor
    rgb, a = _split(img)
    rgb_up = cv2.resize(rgb, (up_size, up_size), interpolation=cv2.INTER_NEAREST)
    a_up = cv2.resize(a, (up_size, up_size), interpolation=cv2.INTER_NEAREST)
    if a_up.ndim == 2:
        a_up = a_up[..., None]
    img_up = _merge(rgb_up, a_up)

    for name, fn, pkey in _PIPELINE:
        if name in _SCALE_GROUP_NAMES and rng.random() < getattr(ranges, pkey):
            img_up = fn(img_up, rng, ranges)
            log.append(f"{name}@{factor}x")

    rgb_down = cv2.resize(img_up[..., :3], (native_size, native_size), interpolation=cv2.INTER_AREA)
    a_down = cv2.resize(img_up[..., 3], (native_size, native_size), interpolation=cv2.INTER_AREA)
    if a_down.ndim == 2:
        a_down = a_down[..., None]
    return _merge(rgb_down, a_down)


def _coerce(sprite: np.ndarray) -> np.ndarray:
    a = np.asarray(sprite)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 3:
        a = np.concatenate([a, np.ones((*a.shape[:2], 1), np.float32)], axis=-1)
    return np.clip(a, 0.0, 1.0)
