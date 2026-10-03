"""optimizer.py — Bộ tối ưu, scheduler và cắt gradient.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch
import torch.optim as optim

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer."""
    if name not in OPTIMIZERS:
        raise ValueError(f"Optimizer '{name}' không nằm trong danh sách hỗ trợ: {OPTIMIZERS}")

    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ cosine (CosineAnnealingLR).

    Trả về None nếu name là None.
    """
    if name is None:
        return None
    if name == "cosine":
        eta_min = kwargs.get("eta_min", 0.0)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    raise ValueError(f"Scheduler không hỗ trợ: {name}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Các bước:
      1. nếu max_norm là None hoặc <= 0: tính chuẩn toàn cục mà không cắt (max_norm=float('inf'))
      2. ngược lại: total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
      3. return float(total_norm)
    """
    params_list = [p for p in params if p.grad is not None]
    if not params_list:
        return 0.0

    if max_norm is None or max_norm <= 0:
        total_norm = torch.nn.utils.clip_grad_norm_(params_list, float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(params_list, max_norm)

    return float(total_norm)
