"""model.py — Định nghĩa kiến trúc mô hình MLP và khởi tạo tham số.

Model: MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}


class MLP(nn.Module):
    """MLP theo quy định chuẩn:
      - Tuyến tính -> ReLU -> [Dropout] cho các lớp ẩn
      - Tuyến tính -> logits cho lớp ra
    """

    def __init__(self, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.init = init
        self.in_features = in_features
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        prev_dim = in_features
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h, bias=True))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h

        # Lớp ra: chuyển về số logits tương ứng số lớp phân loại
        layers.append(nn.Linear(prev_dim, num_classes, bias=True))

        self.net = nn.Sequential(*layers)

        # Khởi tạo trọng số
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias luôn = 0, trừ 'default').

    init:
        "zeros"   : W = 0, bias = 0
        "normal"  : W ~ N(0, 0.01^2), bias = 0
        "xavier"  : nn.init.xavier_normal_ (Var = 2/(n_in+n_out)), bias = 0
        "he"      : nn.init.kaiming_normal_(w, nonlinearity="relu")  (Var = 2/n_in), bias = 0
        "default" : giữ khởi tạo mặc định của PyTorch (kaiming_uniform + bias uniform)
    """
    if init == "default":
        return

    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init == "zeros":
                nn.init.zeros_(m.weight)
            elif init == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Cách khởi tạo không hợp lệ: {init}. "
                                 f"Chọn một trong: 'zeros', 'normal', 'xavier', 'he', 'default'")

            if m.bias is not None:
                nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt sau mỗi lớp kích hoạt ReLU và lớp ra ở bước 0."""
    was_training = model.training
    model.eval()

    stds: list[float] = []
    h = x
    # Duyệt qua các lớp trong sequential net
    for layer in model.net:
        h = layer(h)
        # Ghi nhận std sau mỗi ReLU (lớp ẩn) hoặc Linear cuối cùng
        if isinstance(layer, (nn.ReLU, nn.Linear)) and layer is not model.net[0]:
            stds.append(float(h.std().item()))

    if was_training:
        model.train()

    return stds
