"""data.py — Chuẩn bị và nạp dữ liệu CoverType.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
NUM_CLASSES = 7


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    proc_path = Path(processed_dir)
    train_path = proc_path / "train.npz"
    eval_path = proc_path / "eval.npz"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu processed tại {processed_dir}. "
            "Hãy chạy `python scripts/split_data.py` trước!"
        )

    with np.load(train_path) as tr_data:
        X_train_full = tr_data["X"].astype(np.float32)
        y_train_full = tr_data["y"].astype(np.int64)

    with np.load(eval_path) as ev_data:
        X_eval = ev_data["X"].astype(np.float32)
        y_eval = ev_data["y"].astype(np.int64)
        eval_row_id = ev_data["row_id"].astype(np.int64)

    assert X_train_full.shape == (464_809, 54), f"Shape X_train_full bất thường: {X_train_full.shape}"
    assert y_train_full.shape == (464_809,), f"Shape y_train_full bất thường: {y_train_full.shape}"
    assert X_eval.shape == (116_203, 54), f"Shape X_eval bất thường: {X_eval.shape}"
    assert y_eval.shape == (116_203,), f"Shape y_eval bất thường: {y_eval.shape}"
    assert eval_row_id.shape == (116_203,), f"Shape eval_row_id bất thường: {eval_row_id.shape}"

    assert X_train_full.dtype == np.float32
    assert y_train_full.dtype == np.int64
    assert X_eval.dtype == np.float32
    assert y_eval.dtype == np.int64
    assert eval_row_id.dtype == np.int64

    assert 0 <= y_train_full.min() and y_train_full.max() < NUM_CLASSES
    assert 0 <= y_eval.min() and y_eval.max() < NUM_CLASSES

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Sử dụng train_test_split phân tầng theo nhãn để tỉ lệ phân bố giữa các lớp đồng đều.
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Lý do không được tính trên val/eval:
      Tính trên val/eval gây data leakage (rò rỉ phân phối dữ liệu tập kiểm thử vào mô hình).
    """
    mean = X_tr[:, :N_NUMERIC].mean(axis=0)
    std = X_tr[:, :N_NUMERIC].std(axis=0)
    # Tránh chia cho 0 nếu std = 0
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu còn dùng lại nó.
    """
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)

    # Tách validation từ train (phân tầng theo nhãn)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)

    # Chuẩn hoá 10 đặc trưng liên tục đầu tiên dựa trên thống kê của train (sau khi tách val)
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    # Chuyển dữ liệu sang PyTorch tensor và đưa lên device
    target_device = torch.device(device)
    X_tr_t = torch.from_numpy(X_tr).to(dtype=torch.float32, device=target_device)
    y_tr_t = torch.from_numpy(y_tr).to(dtype=torch.int64, device=target_device)

    X_val_t = torch.from_numpy(X_val).to(dtype=torch.float32, device=target_device)
    y_val_t = torch.from_numpy(y_val).to(dtype=torch.int64, device=target_device)

    X_eval_t = torch.from_numpy(X_eval).to(dtype=torch.float32, device=target_device)
    y_eval_t = torch.from_numpy(y_eval).to(dtype=torch.int64, device=target_device)

    # Thống kê và kiểm tra
    val_counts = np.bincount(y_val, minlength=NUM_CLASSES)
    maj_class = int(np.argmax(val_counts))
    maj_acc = float(val_counts[maj_class] / len(y_val))

    print(f"--- Dữ liệu đã chuẩn bị trên {target_device} ---")
    print(f"Train subset : {X_tr.shape[0]:,d} mẫu")
    print(f"Validation   : {X_val.shape[0]:,d} mẫu")
    print(f"Eval (cuối)  : {X_eval.shape[0]:,d} mẫu")
    print(f"Tỉ lệ đoán lớp đa số (lớp {maj_class}) trên validation: {maj_acc:.4f} (~ {maj_acc*100:.2f}%)")
    print(f"Mean của 10 cột số trên train còn lại: {X_tr[:, :N_NUMERIC].mean(axis=0).round(4)}")
    print(f"Std của 10 cột số trên train còn lại: {X_tr[:, :N_NUMERIC].std(axis=0).round(4)}")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }


def iterate_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                    generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
