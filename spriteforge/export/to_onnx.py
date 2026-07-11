"""
STUB — implement only if C# / Unity Editor port is needed later.
"""

from __future__ import annotations

import torch
from spriteforge.model.config import get_config
from spriteforge.model.vqgan import SpriteVQGAN


def export_to_onnx(checkpoint_path: str, output_onnx_path: str, size: int = 32) -> None:
    """Export PyTorch SpriteVQGAN checkpoint to ONNX format."""
    config = get_config(str(size))
    model = SpriteVQGAN(config)
    model.eval()

    dummy_input = torch.randn(1, 4, size, size)
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    print(f"[+] Successfully exported model to {output_onnx_path}")
