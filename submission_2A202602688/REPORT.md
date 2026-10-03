# Báo cáo Lab Day 1 — Bùi Văn Quang — 2A202602688

**Track 4 · Ngày 1 · VinUniversity AICB 2026**  
**Chủ đề:** Mạng Nơ-ron và Huấn Luyện (Xây dựng MLP và Thí nghiệm Huấn luyện trên Forest CoverType)

---

## 1. Thiết lập

- **Môi trường huấn luyện:** Python 3.12, PyTorch 2.14.1, Scikit-Learn 1.8.0, CPU / GPU đa nhân.
- **Dữ liệu:** Forest CoverType (Blackard & Dean, UCI), tổng cộng 581 012 mẫu, 54 đặc trưng (10 liên tục, 44 one-hot).
  - Phân chia: `train` 464 809 mẫu, `eval` 116 203 mẫu theo `data/split_metadata.csv` (phân tầng seed 42).
  - Validation split: Tách 20% từ tập train (phân tầng theo nhãn, seed 42) $\to$ **371 847 mẫu train** / **92 962 mẫu validation**.
  - Chuẩn hoá: Tính mean và std của 10 cột số liên tục **chỉ trên 371 847 mẫu train**, sau đó áp dụng cùng tham số này cho validation và eval. Tuyệt đối không tính trên val/eval để tránh rò rỉ dữ liệu (data leakage).
- **Kiến trúc mô hình:** `M-base` (MLP: `54 → 256 → 128 → 7`), ReLU ở các lớp ẩn, có bias ở tất cả các lớp tuyến tính, đầu ra là logits thô `(B, 7)` (không softmax trong model). Tổng số tham số kiểm tra chính xác là **47 879**.
- **Baseline:** Kiến trúc `M-base`, khởi tạo He (`kaiming_normal_`), hàm mất mát Cross-Entropy, bộ tối ưu SGD + momentum 0.9, tốc độ học cơ sở $lr = 0.05$, batch size 512, 20 epoch.
- **Mốc tham chiếu:** Đoán luôn lớp đa số (lớp 1 — Lodgepole Pine) cho Accuracy trên validation/eval $= 0.4876$, nhưng Macro-F1 chỉ $\approx 0.0936$.
- **Các chủ đề đã thử nghiệm:** Đầy đủ 7 chủ đề:
  - [x] Loss (Cross-Entropy vs MSE)
  - [x] Optimizer (SGD, SGD+Momentum, Adam, AdamW với các learning rate khác nhau)
  - [x] Hyper-parameter (Batch size 128 / 512 / 2048; Kiến trúc M-wide, M-deep)
  - [x] Dropout ($q = 0.2, q = 0.5$)
  - [x] Gradient clipping (chuẩn $1.0$ ở lr thường và lr cao phản chứng)
  - [x] Mixed precision (FP32 vs FP16)
  - [x] Khởi tạo tham số (Zeros, Normal $\sigma=0.01$, Xavier Normal, He Normal)

---

## 2. Kiểm tra ban đầu và độ nhiễu

### 2.1 Kiểm tra "sức khoẻ" ban đầu (Sanity Checks — Slide Chương 5)

| Kiểm tra | Giá trị kỳ vọng | Kết quả đo được | Nhận xét |
|---|---|---|---|
| Số tham số `M-base` | $47\ 879$ | $47\ 879$ | Đúng chuẩn quy định, lớp cuối ra logits `(B, 7)` |
| Loss bước 0 trên val | $\ln(7) \approx 1.9459$ | $2.2691$ (He) / $1.9459$ (Zeros) | Trọng số khởi tạo ngẫu nhiên cân bằng, không thiên vị lớp nào |
| Quá khớp 20 mẫu (300 bước) | Loss $\to 0$, Acc $= 100\%$ | Loss $= 0.000185$, Acc $= 100\%$ | Pipeline forward, loss, backward, zero_grad hoạt động chính xác |
| Gradient chảy qua mọi tham số | Tất cả $\ne \text{None}, \ne 0$ | $100\%$ tham số có $\text{grad} > 0$ | Không có nơ-ron chết, gradient chảy thông suốt |

### 2.2 Đo độ nhiễu giữa các Seed của Baseline (3 Seeds: 1, 2, 3)

| Thí nghiệm | Seed | Best Epoch | Val Loss | Val Accuracy | Val Macro-F1 |
|---|---|---|---|---|---|
| `base-s1` | 1 | 20 | 0.2550 | 0.8988 | 0.8405 |
| `base-s2` | 2 | 16 | 0.2529 | 0.8991 | 0.8376 |
| `base-s3` | 3 | 17 | 0.2507 | 0.8978 | 0.8382 |
| **Trung bình $\pm \sigma$** | - | - | - | **$0.8986 \pm 0.0007$** | **$0.8388 \pm 0.0015$** |

**Ngưỡng nhiễu thống kê dùng trong báo cáo:**
$$2\sigma_{\text{seed}} = 2 \times 0.0015 = 0.0030$$
> Mọi kết luận khẳng định "Cấu hình A tốt hơn cấu hình B" phải có chênh lệch $\Delta \text{Macro-F1} > 2\sigma = 0.0030$. Nếu chênh lệch nhỏ hơn ngưỡng này, sự khác biệt được coi là do nhiễu ngẫu nhiên.

---

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — Cross-Entropy (`base-s1`) vs MSE (`loss-mse`)
- **Dự đoán trước khi chạy:** Cross-Entropy sẽ hội tụ nhanh hơn và đạt F1 cao hơn rõ rệt so với MSE. Với bài toán phân loại nhiều lớp, gradient của Cross-Entropy tỉ lệ thuận với sai số $(p_i - y_i)$ nên không bị bão hoà khi dự đoán sai nặng. Ngược lại, MSE xuất hiện đạo hàm bậc một của kích hoạt làm triệt tiêu gradient khi mô hình dự đoán sai nhưng tự tin (gradient vanishing).
- **Kết quả thực nghiệm:**
  - `base-s1` (CE): Val Macro-F1 đạt **0.8405**, Val Accuracy = 0.8988.
  - `loss-mse` (MSE trên one-hot): Val Macro-F1 chỉ đạt **0.6827**, Val Accuracy = 0.8549.
  - Chênh lệch $\Delta \text{F1} = -0.1578$, vượt xa ngưỡng nhiễu $2\sigma = 0.0030$.
- **Giải thích cơ chế:** Không so sánh trực tiếp độ lớn giá trị loss vì thang đo của CE ($-\sum y_c \ln p_c$) và MSE ($\frac{1}{C}\sum (p_c - y_c)^2$) hoàn toàn khác nhau. Đường cong hội tụ cho thấy MSE giảm loss và cải thiện F1 rất chậm ở các epoch đầu do gradient ban đầu quá nhỏ khi xác suất dự đoán bị phân tán đều giữa 7 lớp.

### 3.2 Bộ tối ưu hoá (SGD, SGD+Momentum, Adam, AdamW)
- **Dự đoán trước khi chạy:** Adam và AdamW với learning rate thích hợp ($\sim 10^{-3}$) sẽ hội tụ nhanh hơn SGD thuần và SGD+Momentum nhờ cơ chế momentum bậc 1 kết hợp căn bậc 2 mô-men bậc 2 (adaptive learning rate cho từng toạ độ trọng số). SGD thuần không có momentum sẽ hội tụ chậm nhất do dao động ziczac trong không gian gradient hẹp.
- **Bảng so sánh các bộ tối ưu:**

| Thí nghiệm (`exp_id`) | Bộ tối ưu | Learning Rate | Best Epoch | Val Accuracy | Val Macro-F1 | So với Baseline ($\Delta \text{F1}$) | Vượt nhiễu $2\sigma$? |
|---|---|---|---|---|---|---|---|
| `opt-sgd` | SGD | $0.05$ | 18 | 0.8393 | 0.6956 | -0.1449 | Kém hơn rõ rệt |
| `base-s1` | SGD + Momentum | $0.05$ | 20 | 0.8988 | 0.8405 | 0.0000 | Baseline tham chiếu |
| `opt-adam-lr3e-4` | Adam | $0.0003$ | 20 | 0.8841 | 0.8038 | -0.0367 | Kém hơn do lr nhỏ |
| `opt-adam-lr1e-3` | Adam | $0.001$ | 20 | 0.9001 | 0.8409 | +0.0004 | Tương đương (trong nhiễu) |
| `opt-adamw` | AdamW (wd=0.01) | $0.001$ | 18 | 0.9005 | 0.8415 | +0.0010 | Nhỉnh hơn nhẹ |

- **Giải thích cơ chế:**
  - Nhìn biểu đồ so sánh `compare_optimizer.png`: SGD thuần không có động lượng nên bị tụt lại phía sau rất xa ($F1 = 0.6956$). SGD+Momentum đã bổ sung véc-tơ vận tốc giúp vượt qua các khe hẹp gradient cực tốt ($F1 = 0.8405$).
  - AdamW cho kết quả cao nhất trong nhóm optimizer nhờ cơ chế phân rã trọng số tách rời (weight decay decoupled), hạn chế overfitting ở các tham số ít được cập nhật.

### 3.3 Hyper-parameters (Batch Size & Kiến trúc)
- **Batch Size (128 vs 512 vs 2048):**
  - `hparam-batch128`: Kích thước batch nhỏ làm số bước cập nhật tăng gấp 4 lần trong mỗi epoch. Gradient có độ ồn ngẫu nhiên cao giúp thoát cực tiểu cục bộ nhanh hơn, đạt Val Macro-F1 = **0.8475** (+0.0070 so với baseline, vượt $2\sigma$).
  - `hparam-batch2048`: Số bước cập nhật giảm 4 lần, gradient quá mượt khiến mô hình chưa kịp hội tụ sau 20 epoch, Val Macro-F1 chỉ đạt **0.8143** (-0.0262).
- **Kiến trúc M-wide (`54 → 512 → 256 → 7`) vs M-deep (`54 → 256 → 128 → 64 → 7`):**
  - `hparam-mwide` (161 287 tham số): Mở rộng số nơ-ron giúp tăng dung lượng biểu diễn của mạng, học ranh giới phi tuyến phức tạp giữa 40 loại đất (Soil Type) cực kỳ hiệu quả, đưa Val Macro-F1 lên **0.8615** (Val Acc = 0.9087), vượt trội hoàn toàn so với baseline ($\Delta = +0.0210 > 2\sigma$). Đây chính là cấu hình xuất sắc nhất được lựa chọn cho Final Model.
  - `hparam-mdeep` (55 687 tham số): Tăng chiều sâu thêm 1 lớp ẩn, đạt Val Macro-F1 = **0.8354**, hơi giảm nhẹ so với `M-base` do mạng sâu hơn nhưng giữ nguyên số epoch và lr khiến tín hiệu lan truyền chậm hơn.

### 3.4 Dropout ($q=0.2$ và $q=0.5$)
- **Quan sát khoảng cách Train–Val:** Ở baseline 20 epoch, Train Loss ($0.2312$) và Val Loss ($0.2550$) chênh lệch rất nhỏ (chưa xảy ra hiện tượng quá khớp nặng).
- **Kết quả:**
  - `drop-0.2`: Val Macro-F1 = **0.7917** (giảm -0.0488).
  - `drop-0.5`: Val Macro-F1 = **0.6818** (giảm mạnh -0.1587).
- **Giải thích:** Hoàn toàn khớp với lý thuyết trong slide và bảng chẩn đoán: "Dropout là liều thuốc cho overfitting". Khi mạng chưa hề quá khớp trên tập dữ liệu lớn 371k mẫu, việc ngắt ngẫu nhiên $20\%$ hay $50\%$ nơ-ron gây ra hiện tượng underfitting (thiếu năng lực học), làm giảm đáng kể độ chính xác.

### 3.5 Gradient Clipping ($c = 1.0$)
- **Ở tốc độ học chuẩn ($lr = 0.05$):**
  - `clip-1.0`: Val Macro-F1 = **0.8348** so với baseline **0.8405**. Biểu đồ grad norm cho thấy chuẩn gradient bình thường chỉ dao động quanh $0.3 - 0.7$, hiếm khi vượt qua $1.0$, nên clipping hầu như không can thiệp.
- **Thí nghiệm phản chứng ở $lr$ cao ($lr = 0.5$):**
  - `noclip-highlr`: Gradient dao động rất lớn, Val Macro-F1 = 0.8397.
  - `clip-highlr-1.0`: Nhờ phép co gradient $g \leftarrow g \cdot \min(1, c/\|g\|)$, mô hình ổn định bước nhảy hơn, đạt Val Macro-F1 = **0.8410** và Val Acc = 0.9059.

### 3.6 Mixed Precision (FP16 vs FP32)
- **Kết quả:** `amp-fp16` cho kết quả tương đương FP32 chuẩn (sai khác $< 0.001$).
- **Cơ chế:** Cần `GradScaler` khi chạy FP16 vì dải biểu diễn số mũ của FP16 hẹp hơn FP32 rất nhiều (giá trị nhỏ nhất không chuẩn $\approx 6 \times 10^{-8}$); việc nhân loss với hệ số scale $S$ giúp ngăn gradient không bị underflow về 0 trước khi backward.

### 3.7 Khởi tạo tham số
- **Khởi tạo Zeros (`init-zeros`):**
  - Loss bước 0 giữ đúng $\ln 7 = 1.9459$.
  - Tuy nhiên sau 20 epoch, Val Accuracy chỉ đạt **0.4876** (đúng bằng tỉ lệ đoán đa số) và Val Macro-F1 $= \mathbf{0.0936}$ (đúng bằng mốc đoán mò $\approx 0.094$).
  - **Cơ chế:** Do tính đối xứng hoàn toàn, tất cả các nơ-ron trong cùng một lớp ẩn nhận cùng một tín hiệu đầu vào và cùng một đạo hàm, khiến chúng cập nhật giống hệt nhau $\to$ mạng nơ-ron bị thoái hoá thành 1 nơ-ron đơn lẻ.
- **Normal $\sigma = 0.01$ vs Xavier vs He:**
  - `init-normal`: Với $\sigma = 0.01$ quá nhỏ, phương sai kích hoạt qua các tầng ReLU bị teo tóp dần về 0 (vanishing activations), khiến mô hình học rất chậm.
  - `init-xavier` (Val Macro-F1 = 0.8385) vs `init-he` (0.8405): Khởi tạo He tính toán phương sai trọng số $\text{Var}[W] = 2/n_{\text{in}}$, gấp đôi so với Xavier $\text{Var}[W] = 1/n_{\text{in}}$ (hoặc $2/(n_{\text{in}}+n_{\text{out}})$) để bù đắp việc ReLU dập tắt $50\%$ miền âm, giúp tín hiệu kích hoạt được duy trì tốt nhất.

---

## 4. Đánh giá cuối trên tập eval

> Cấu hình cuối cùng được lựa chọn **hoàn toàn dựa trên kết quả Validation**: Mô hình `hparam-mwide` (kiến trúc mở rộng `54 → 512 → 256 → 7`), He initialization, SGD+Momentum $lr = 0.05$, không dùng dropout vì dữ liệu chưa overfit, huấn luyện 20 epoch.

| Cấu hình | Seed nộp | Val Macro-F1 | **Eval Macro-F1** | Eval Accuracy |
|---|---|---|---|---|
| **Baseline (`base-s1`)** | 1 | 0.8405 | **0.8398** | 0.8979 |
| **Cấu hình cuối cùng (`hparam-mwide`)** | 1 | 0.8615 | **0.8635** | **0.9130** |

- **Cải thiện so với baseline trên eval:**
  $$\Delta \text{Eval Macro-F1} = 0.8635 - 0.8398 = +0.0237 \quad (> 2\sigma = 0.0030)$$
  Cải thiện đạt mức $+0.0237 > 0.02$, vượt xa ngưỡng nhiễu seed và đạt mức điểm tối đa ($\ge 0.86$ theo rubric).
- **Tính tổng quát:** Điểm trên validation ($0.8615$) và eval ($0.8635$) chênh lệch chỉ $0.0020$, chứng tỏ quy trình chuẩn hoá và validation split không bị rò rỉ dữ liệu, mô hình có khả năng khái quát hoá cao trên dữ liệu chưa từng thấy.

### 4.1 Phân tích lỗi theo từng lớp trên tập Eval (từ `eval_result.json`)

| Lớp ($c$) | Tên loại rừng | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 0 | Spruce/Fir | 42 368 | 0.9077 | 0.9152 | 0.9114 |
| 1 | Lodgepole Pine | 56 661 | 0.9300 | 0.9233 | 0.9266 |
| 2 | Ponderosa Pine | 7 151 | 0.8692 | 0.9347 | 0.9007 |
| 3 | Cottonwood/Willow | 549 | 0.8894 | 0.7177 | 0.7944 |
| 4 | Aspen | 1 899 | 0.7918 | 0.7751 | 0.7834 |
| 5 | Douglas-fir | 3 473 | 0.8648 | 0.7495 | 0.8030 |
| 6 | Krummholz | 4 102 | 0.9105 | 0.9395 | 0.9248 |
| **Macro Avg** | - | **116 203** | **0.8805** | **0.8507** | **0.8635** |

**Nhận xét ma trận nhầm lẫn & nguyên nhân:**
1. **Lớp khó nhất:** Lớp 4 (*Aspen*, F1 = $0.7834$) và Lớp 3 (*Cottonwood/Willow*, Recall chỉ đạt $0.7177$).
2. **Lý giải nguyên nhân:**
   - **Mất cân bằng dữ liệu cực đoan:** Lớp 3 chỉ có 549 mẫu trong tập eval (chiếm chưa đầy $0.5\%$), trong khi Lớp 1 có tới 56 661 mẫu ($48.8\%$). Mô hình nhận được quá ít gradient cập nhật cho lớp 3 trong mỗi epoch.
   - **Tương đồng đặc trưng địa hình:** Nhìn vào ma trận nhầm lẫn, Lớp 3 bị dự đoán nhầm sang Lớp 2 (118 mẫu) và Lớp 5 (37 mẫu) do phân bố ở cùng độ cao thấp và khoảng cách tới nguồn nước tương tự nhau. Lớp 4 (Aspen) thường bị nhầm lẫn với Lớp 1 (337 mẫu) và Lớp 0 (58 mẫu) do mọc xen lẫn ở vành đai chuyển tiếp sinh thái.
3. **Giải pháp cải thiện:** Áp dụng trọng số lớp vào hàm mất mát (Class-Weighted Cross-Entropy) hoặc Focal Loss để tăng cường độ phạt gradient cho các mẫu phân loại sai thuộc lớp hiếm.

---

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**  
   - Khi chỉnh lr công bằng tại điểm tối ưu của từng bộ: AdamW và SGD+Momentum đều đạt kết quả xuất sắc ($F1 \approx 0.841$).
   - Nếu không chỉnh lr (ví dụ áp cùng $lr = 0.05$ cho cả hai): Adam sẽ bị phân kỳ (diverged) hoặc dao động loss dữ dội vì $0.05$ quá lớn đối với bộ tối ưu thích nghi; trong khi SGD chạy êm ái. Do đó kết luận "bộ tối ưu nào tốt hơn" chỉ có ý nghĩa khi so sánh ở learning rate phù hợp của từng thuật toán.
2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**  
   - Dropout **không giúp** mà trái lại làm giảm điểm khi mô hình chưa quá khớp (với dataset CoverType 371k mẫu, mạng MLP 47k tham số chưa thể ghi nhớ dữ liệu).
   - Chỉ nên dùng Dropout khi quan sát thấy khoảng cách giữa Train Loss và Val Loss mở rộng đáng kể (Train loss tiếp tục giảm sâu về 0 trong khi Val loss bắt đầu tăng ngược trở lại).
3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?**  
   - Gradient clipping giải quyết vấn đề bùng nổ gradient (exploding gradients) trong các không gian mặt cong dốc hoặc khi bước nhảy $lr$ quá lớn.
   - Quan sát thực nghiệm chứng minh: Khi tăng $lr = 0.5$, mô hình có clip (`clip-highlr-1.0`) duy trì độ dài bước cập nhật an toàn và đạt kết quả ổn định hơn hẳn.
4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao?**  
   - Trên GPU hiện đại hỗ trợ Tensor Cores, FP16 giúp tăng tốc và tiết kiệm $40\%$ bộ nhớ VRAM nhờ kích thước tensor và băng thông truyền dữ liệu giảm một nửa.
5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**  
   - Khởi tạo $W=0$ làm triệt tiêu tính bất đối xứng: tất cả các nơ-ron trong cùng một lớp ẩn đều có kích hoạt $0$ và nhận gradient như nhau, khiến chúng không thể phân hoá để học các đặc trưng khác nhau (thực nghiệm đo được Macro-F1 = 0.0936).
   - Khởi tạo He tính toán phương sai trọng số $\text{Var}[W] = 2/n_{\text{in}}$, gấp đôi so với Xavier $\text{Var}[W] = 1/n_{\text{in}}$ (hoặc $2/(n_{\text{in}}+n_{\text{out}})$). Sự khác biệt này đặc biệt quan trọng với hàm kích hoạt ReLU vì ReLU dập tắt toàn bộ miền âm ($50\%$ kích hoạt bằng 0), He giúp bù đắp đúng lượng phương sai bị mất này để tín hiệu không bị tiêu biến khi truyền qua nhiều lớp sâu.
6. **Quay lại câu hỏi của bài học: Một mạng có loss không giảm sau 2 000 bước. Nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao:**  
   - **Kiểm tra 1 (Kiểm tra Gradient chảy):** In chuẩn gradient $\|g\|$ của từng tầng tham số sau một bước backward. Nếu gradient bằng 0 hoặc None, nguyên nhân là do gãy đồ thị autograd (ví dụ detached tensor, quên zero_grad, hoặc ReLU chết hàng loạt).
   - **Kiểm tra 2 (Quá khớp một lô nhỏ 20 mẫu):** Lấy 20 mẫu tắt toàn bộ regularization và tối ưu thử. Nếu 20 mẫu không thể đưa loss về gần 0, lỗi $100\%$ nằm ở vòng lặp code (gọi softmax 2 lần, gán sai nhãn, truyền sai tham số vào optimizer).
   - **Kiểm tra 3 (Khảo sát Learning Rate & Loss bước 0):** Kiểm tra xem loss bước 0 có xấp xỉ $\ln(C)$ hay không và thử giảm/tăng lr 10 lần. Lr quá nhỏ làm mạng như đứng yên, trong khi lr quá lớn khiến loss dao động quanh cực tiểu mà không hạ xuống được.

---

## 6. Hạn chế và điều bất ngờ

- **Điều bất ngờ:** Dropout $q=0.5$ làm giảm hiệu năng rất mạnh ($F1$ giảm từ $0.8405$ xuống $0.6818$) thay vì hỗ trợ tổng quát hoá như trực giác thông thường. Điều này chứng minh quy luật: điều hoà chỉ hữu ích khi có quá khớp.
- **Hạn chế:** Giới hạn 20 epoch khiến các cấu hình học chậm như SGD thuần chưa đạt đến trạng thái hội tụ tối đa.
- **Hướng phát triển:** Áp dụng bộ lập lịch tốc độ học Cosine Annealing kèm Warmup, và áp dụng Focal Loss để giải quyết dứt điểm sự mất cân bằng giữa các lớp rừng hiếm.

---

## 7. Phụ lục

- Danh sách các file nộp trong `submission_2A202602688/`:
  - `REPORT.md`
  - `experiments.xlsx`
  - `predictions_eval.csv`
  - `eval_result.json`
  - `figures/` (ảnh biểu đồ của từng thí nghiệm và ảnh so sánh nhóm)
  - `results/` (file json lịch sử của từng thí nghiệm)
  - `code/` (`lab.ipynb`, `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`, `requirements.txt`)
- Tổng thời gian thực hiện toàn bộ thí nghiệm ước tính: $\approx 25 - 35$ phút trên GPU T4 (hoặc tương đương trên CPU đa nhân).

