"""train.py — Vòng lặp huấn luyện, đánh giá, dự đoán và thí nghiệm.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` mặc định hợp lý (0.05) được kiểm chứng trên val.
DEFAULT_CFG = dict(
    exp_id="base-s1",
    group="baseline",
    description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # tốc độ học cơ sở
    weight_decay=0.0,
    momentum=0.9,
    batch=512,
    epochs=20,
    hidden=(256, 128),
    dropout=0.0,
    init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    was_training = model.training
    model.eval()

    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))

    if was_training:
        model.train()

    return torch.cat(preds, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Hàm mất mát không hỗ trợ: {loss_name}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    was_training = model.training
    model.eval()

    n = len(X)
    total_loss = 0.0
    total_correct = 0
    cm = np.zeros((7, 7), dtype=np.int64)

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)

        if loss_name == "ce":
            loss_b = F.cross_entropy(logits, yb, reduction="sum")
        else:
            yb_onehot = F.one_hot(yb, num_classes=logits.shape[1]).to(dtype=logits.dtype)
            loss_b = F.mse_loss(logits, yb_onehot, reduction="sum")

        total_loss += float(loss_b.item())
        pred_b = logits.argmax(dim=1)
        total_correct += int((pred_b == yb).sum().item())

        y_true_np = yb.cpu().numpy()
        y_pred_np = pred_b.cpu().numpy()
        np.add.at(cm, (y_true_np, y_pred_np), 1)

    if was_training:
        model.train()

    avg_loss = total_loss / n
    acc = total_correct / n
    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "cm": cm,
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg["seed"])

    device = data["X_tr"].device
    hidden = tuple(cfg["hidden"])
    dropout = float(cfg.get("dropout", 0.0))
    init = cfg.get("init", "he")

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    expected = EXPECTED_PARAMS.get(hidden)
    if expected is not None:
        assert count_params(model) == expected, (
            f"Số tham số {count_params(model)} không khớp quy định {expected}"
        )

    optimizer = build_optimizer(
        name=cfg["optimizer"],
        params=model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )

    precision = cfg.get("precision", "fp32")
    use_cuda = device.type == "cuda"
    use_amp = precision in ("fp16", "bf16") and use_cuda
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (precision == "fp16" and use_cuda) else None

    # Tập con cố định để đo train loss ở eval mode (50 000 mẫu đầu của X_tr)
    eval_tr_size = min(50_000, len(data["X_tr"]))
    X_tr_eval = data["X_tr"][:eval_tr_size]
    y_tr_eval = data["y_tr"][:eval_tr_size]

    # Bước 0: đo val loss TRƯỚC bước cập nhật đầu tiên
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_res["loss"]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    diverged = False

    generator = torch.Generator(device=device).manual_seed(cfg["seed"])

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, cfg["epochs"] + 1):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.train()
        batch_grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], cfg["batch"], generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), cfg.get("clip_norm"))
                batch_grad_norms.append(gn)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), cfg.get("clip_norm"))
                batch_grad_norms.append(gn)
                optimizer.step()

        if diverged:
            print(f"[{cfg['exp_id']}] CẢNH BÁO: Bị phân kỳ (NaN/Inf) tại epoch {epoch}!")
            break

        if use_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        # Đánh giá cuối epoch ở chế độ eval
        tr_eval = evaluate(model, X_tr_eval, y_tr_eval, loss_name=cfg["loss"])
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        mean_gn = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(epoch_time)

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # Tổng kết
    if not diverged and best_state is not None:
        idx = best_epoch - 1
        final_tr_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        best_val_acc = history["val_acc"][idx]
        best_val_macro_f1 = history["val_macro_f1"][idx]
    else:
        final_tr_loss = None
        final_val_loss = None
        best_val_acc = 0.0
        best_val_macro_f1 = 0.0

    peak_mem = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if use_cuda else 0.0
    )
    time_per_epoch = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    summary = {
        "exp_id": cfg["exp_id"],
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_tr_loss) if final_tr_loss is not None else None,
        "final_val_loss": float(final_val_loss) if final_val_loss is not None else None,
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(time_per_epoch),
        "peak_mem_MB": float(peak_mem),
        "diverged": bool(diverged),
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id.astype(int), "pred": preds.astype(int)})
    df.to_csv(out_file, index=False)
    print(f"Đã ghi dự đoán eval vào {out_file} ({len(df):,d} dòng).")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Nạp best_state, dự đoán eval, ghi predictions_eval.csv."""
    device = data["X_eval"].device
    hidden = tuple(cfg["hidden"])
    dropout = float(cfg.get("dropout", 0.0))

    model = MLP(hidden=hidden, dropout=dropout, in_features=54, num_classes=7).to(device)
    model.load_state_dict(result["best_state"])

    preds = predict(model, data["X_eval"], batch_size=8192)
    preds_np = preds.cpu().numpy()
    write_predictions(data["eval_row_id"], preds_np, pred_path)
