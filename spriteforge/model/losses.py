"""
Loss functions for VQ-GAN training.

Why replace standard LPIPS at 16px/32px?
Standard LPIPS uses VGG-16 trained on 224x224 ImageNet photos. Feeding a 16x16 sprite into VGG
either fails due to receptive field mismatch or requires bilinear upsampling to 224x224, which
destroys pixel-art edge definitions.

Instead, we use:
1. Sobel/Laplacian Edge-Difference Loss: Penalizes blurry edges and rewards 1-pixel outlines.
2. Premultiplied RGB Loss + Alpha BCE: Keeps background color noise from polluting gradients.
3. Feature Matching Loss: Uses intermediate layers of our PatchDiscriminator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    """Computes L1 difference between Sobel edge gradients of prediction and target."""
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute edges on grayscale luminance or per-channel
        pred_lum = pred[..., :3, :, :].mean(dim=1, keepdim=True)
        target_lum = target[..., :3, :, :].mean(dim=1, keepdim=True)

        pred_grad_x = F.conv2d(pred_lum, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred_lum, self.kernel_y, padding=1)
        target_grad_x = F.conv2d(target_lum, self.kernel_x, padding=1)
        target_grad_y = F.conv2d(target_lum, self.kernel_y, padding=1)

        loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        return loss


class SpriteReconstructionLoss(nn.Module):
    """Combined L1 + Edge + Alpha + Background Suppression loss tailored for sprites."""
    def __init__(self, edge_weight: float = 2.0, alpha_weight: float = 1.0, color_weight: float = 5.0, bg_weight: float = 2.0):
        super().__init__()
        self.edge_loss = SobelEdgeLoss()
        self.edge_weight = edge_weight
        self.alpha_weight = alpha_weight
        self.color_weight = color_weight
        self.bg_weight = bg_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] == 4 and target.shape[1] == 4:
            pred_rgb, pred_a = pred[:, :3], pred[:, 3:4]
            target_rgb, target_a = target[:, :3], target[:, 3:4]

            # Premultiply RGB by target alpha so transparent backgrounds don't penalize color
            l1_rgb = F.l1_loss(pred_rgb * target_a, target_rgb * target_a)
            loss_alpha = F.binary_cross_entropy(pred_a, target_a)
            loss_edge = self.edge_loss(pred_rgb * target_a, target_rgb * target_a)

            # Explicit background suppression: penalize color and alpha where target_a == 0
            bg_mask = 1.0 - target_a
            loss_bg_rgb = F.l1_loss(pred_rgb * bg_mask, torch.zeros_like(pred_rgb))
            loss_bg_a = F.l1_loss(pred_a * bg_mask, torch.zeros_like(pred_a))
            loss_bg = loss_bg_rgb + loss_bg_a

            return self.color_weight * l1_rgb + self.alpha_weight * loss_alpha + self.edge_weight * loss_edge + self.bg_weight * loss_bg
        else:
            return self.color_weight * F.l1_loss(pred, target) + self.edge_weight * self.edge_loss(pred, target)


class OrthogonalRegularization(nn.Module):
    """Frobenius-norm penalty measuring how far codebook rows are from mutual orthogonality.

    Used as a MONITORING METRIC ONLY. The codebook is an EMA-updated buffer (no gradients),
    so this value must not be added to the training loss — compute it under torch.no_grad().
    Lower = codes more orthogonal (healthy vocabulary); higher = dimensional collapse.
    """
    def __init__(self):
        super().__init__()

    def forward(self, codebook_weight: torch.Tensor) -> torch.Tensor:
        w = F.normalize(codebook_weight, p=2, dim=1)
        corr = torch.matmul(w, w.t())
        identity = torch.eye(corr.size(0), device=corr.device)
        return torch.norm(corr - identity, p="fro")
