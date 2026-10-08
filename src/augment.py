"""
Physically consistent augmentations for UCI HAR windows in raw units (PROTOCOL.md section 5).

Channel layout (src.dataset.SIGNAL_NAMES): total_acc 0-2, body_acc 3-5, body_gyro 6-8.
total_acc = body_acc + gravity, with gravity estimated per sample as total - body. Every
augmentation edits body/gyro/gravity and rebuilds total, so the identity is preserved:
  scaling  - body_acc and gyro times one factor per window (movement intensity)
  jitter   - noise added to body_acc (and therefore to total) and, independently, to gyro
  rotation - one small 3-D rotation applied jointly to body, gravity and gyro (placement change);
             full SO(3) is excluded because gravity direction separates the static postures.
"""
import math
from dataclasses import dataclass

import torch

STRENGTHS = {
    "weak": dict(scale_sigma=0.1, jitter_sigma=0.05, max_degrees=10.0),
    "strong": dict(scale_sigma=0.2, jitter_sigma=0.1, max_degrees=30.0),
}


@dataclass
class AugConfig:
    scale_sigma: float = 0.1
    jitter_sigma: float = 0.05
    max_degrees: float = 10.0
    rotate: bool = True
    scale: bool = True
    jitter: bool = True

    @classmethod
    def named(cls, name: str, **overrides) -> "AugConfig":
        return cls(**{**STRENGTHS[name], **overrides})


def random_rotations(n: int, max_degrees: float, generator=None, device=None) -> torch.Tensor:
    """[n, 3, 3] rotations about uniformly random axes with angle uniform in [-max, max] (Rodrigues)."""
    axis = torch.randn(n, 3, generator=generator, device=device)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
    angle = (torch.rand(n, generator=generator, device=device) * 2 - 1) * math.radians(max_degrees)
    K = torch.zeros(n, 3, 3, device=device)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s, c = torch.sin(angle).view(n, 1, 1), torch.cos(angle).view(n, 1, 1)
    eye = torch.eye(3, device=device).expand(n, 3, 3)
    return eye + s * K + (1 - c) * (K @ K)


def augment(x: torch.Tensor, cfg: AugConfig, channel_std: torch.Tensor, generator=None) -> torch.Tensor:
    """x: [B, T, 9] raw windows -> augmented copy (same shape, total = body + gravity kept)."""
    B, dev = x.size(0), x.device
    body, gravity, gyro = x[..., 3:6], x[..., 0:3] - x[..., 3:6], x[..., 6:9]
    if cfg.scale:
        s = 1 + cfg.scale_sigma * torch.randn(B, 1, 1, generator=generator, device=dev)
        body, gyro = body * s, gyro * s
    if cfg.jitter:
        body = body + cfg.jitter_sigma * channel_std[3:6] * torch.randn(body.shape, generator=generator, device=dev)
        gyro = gyro + cfg.jitter_sigma * channel_std[6:9] * torch.randn(gyro.shape, generator=generator, device=dev)
    if cfg.rotate and cfg.max_degrees > 0:
        R = random_rotations(B, cfg.max_degrees, generator, dev).transpose(1, 2)  # row vectors: v @ R^T
        body, gravity, gyro = body @ R, gravity @ R, gyro @ R
    return torch.cat([body + gravity, body, gyro], dim=-1)
