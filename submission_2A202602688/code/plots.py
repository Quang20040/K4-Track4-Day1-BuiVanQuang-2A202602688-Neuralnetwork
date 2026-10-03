"""plots.py — Vẽ biểu đồ huấn luyện từng thí nghiệm và biểu đồ so sánh.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    hist = result["history"]
    sum_info = result["summary"]
    epochs = hist["epoch"]

    if not epochs:
        return

    best_epoch = sum_info.get("best_epoch", -1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (1) Loss
    ax1 = axes[0]
    ax1.plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", marker="o", markersize=3)
    ax1.plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", marker="s", markersize=3)
    if best_epoch > 0:
        ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    ax1.set_title("Loss vs Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # (2) Val Metrics
    ax2 = axes[1]
    ax2.plot(epochs, hist["val_acc"], label="Val Accuracy", color="#2ca02c", marker="o", markersize=3)
    if "val_macro_f1" in hist and hist["val_macro_f1"]:
        ax2.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#d62728", marker="^", markersize=3)
    if best_epoch > 0:
        ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    ax2.set_title("Validation Accuracy & Macro-F1", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Score")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # (3) Gradient Norm (trước khi clip)
    ax3 = axes[2]
    ax3.plot(epochs, hist["grad_norm"], label="Global Grad Norm", color="#9467bd", marker="x", markersize=3)
    clip_c = cfg.get("clip_norm")
    if clip_c is not None and clip_c > 0:
        ax3.axhline(clip_c, color="red", linestyle="--", alpha=0.8, label=f"Clip Threshold ({clip_c})")
    ax3.set_title("Grad Norm (Before Clip)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("L2 Norm")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    title_str = (
        f"Exp: {cfg.get('exp_id', '')} | Opt: {cfg.get('optimizer')} (lr={cfg.get('lr')}) | "
        f"Batch: {cfg.get('batch')} | Loss: {cfg.get('loss')} | Init: {cfg.get('init')}"
    )
    fig.suptitle(title_str, fontsize=12, fontweight="bold", y=1.03)

    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số ('val_loss', 'val_macro_f1', 'grad_norm', ...) của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường.
    """
    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "exp")
        epochs = hist.get("epoch", [])
        values = hist.get(metric, [])

        if epochs and values and len(epochs) == len(values):
            ax.plot(epochs, values, marker="o", markersize=3, label=f"{exp_id}")

    ax.set_title(title or f"So sánh {metric} giữa các thí nghiệm", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
