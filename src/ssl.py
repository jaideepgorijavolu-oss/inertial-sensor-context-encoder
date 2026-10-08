"""
SimCLR-style contrastive pretraining of the existing SensorEncoder (Chen et al., 2020, NT-Xent).
Inputs are unlabeled raw pool windows only; nothing in this module accepts labels.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.augment import AugConfig, augment
from src.models import SensorEncoder


class ProjectionHead(nn.Module):
    """256 -> 256 -> 128 MLP; discarded after pretraining (features are taken before it)."""

    def __init__(self, in_dim: int = 256, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, h):
        return self.net(h)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Normalized-temperature cross entropy. Row i of z1 and row i of z2 are the positive pair; the
    other 2N-2 views in the batch are negatives. Mean over all 2N anchors.
    """
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


@torch.no_grad()
def collapse_stats(h: torch.Tensor) -> dict:
    """Representation health on a fixed batch: per-dim std of L2-normalized features (~1/sqrt(d)
    when spread out, ~0 when collapsed) and effective rank (exp entropy of singular values)."""
    zn = F.normalize(h.float(), dim=1)
    sv = torch.linalg.svdvals(h.float() - h.float().mean(0))
    p = sv / sv.sum().clamp_min(1e-12)
    erank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))
    return {"norm_std": float(zn.std(0).mean()), "effective_rank": erank, "dim": h.size(1)}


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / (std + 1e-6)


def pretrain(X_raw: np.ndarray, mean, std, channel_std, aug: AugConfig, tau: float, seed: int, device,
             epochs: int = 200, batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-4,
             log=print) -> tuple:
    """
    Contrastive pretraining on unlabeled raw windows. Checkpoint rule (predeclared): final epoch.
    Returns (encoder state_dict, history list).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    gcpu = torch.Generator().manual_seed(seed)
    X = torch.tensor(X_raw, dtype=torch.float32, device=device)
    mean, std, cstd = (torch.as_tensor(np.asarray(a).reshape(-1), dtype=torch.float32, device=device)
                       for a in (mean, std, channel_std))
    encoder, head = SensorEncoder().to(device), ProjectionHead().to(device)
    params = [*encoder.parameters(), *head.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(X) // batch_size)            # drop last partial batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    probe = X[torch.randperm(len(X), generator=gcpu)[:1024].to(device)]
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        perm = torch.randperm(len(X), generator=gcpu).to(device)
        total = 0.0
        for k in range(steps_per_epoch):
            xb = X[perm[k * batch_size:(k + 1) * batch_size]]
            v1 = standardize(augment(xb, aug, cstd, g), mean, std)
            v2 = standardize(augment(xb, aug, cstd, g), mean, std)
            z = head(encoder(torch.cat([v1, v2])))
            loss = nt_xent(z[:len(xb)], z[len(xb):], tau)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite contrastive loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            encoder.eval()
            with torch.no_grad():
                stats = collapse_stats(encoder(standardize(probe, mean, std)))
            history.append({"epoch": epoch, "loss": total / steps_per_epoch, **stats,
                            "elapsed_s": time.perf_counter() - t0})
            log(f"  [ssl tau={tau}] epoch {epoch:03d} loss {total / steps_per_epoch:.4f} "
                f"norm_std {stats['norm_std']:.4f} erank {stats['effective_rank']:.1f}")
            if stats["norm_std"] < 0.1 / np.sqrt(stats["dim"]):
                history[-1]["collapse_flag"] = True
    state = {k: v.detach().to("cpu", copy=True) for k, v in encoder.state_dict().items()}
    return state, history
