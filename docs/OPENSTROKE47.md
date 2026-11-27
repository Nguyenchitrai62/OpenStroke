# OpenStroke47: FIRE 3 lớp (`other` / `main` / `branch`) trên kiến trúc OpenStroke46

Giả thuyết: H176 (E152), E152c (đối chứng chỉ GT, chưa chạy). Code: kiến trúc của [OPENSTROKE46](OPENSTROKE46.md); đầu ra K lớp
(`stroke_classes=auto`, `loss_multiclass`, `lovasz_softmax_flat` trong `openstroke/models/openstroke2.py`), `class_names` đọc từ
`manifest.json`, metric 3 lớp trong `openstroke/metrics.py`. Test: `tests/test_os47_fire3.py`. Model: `model=openstroke47`
(bí danh `openstroke_fire3`; cùng mạng `openstroke46`, khác công thức train).

## 1. Bằng chứng

Khảo sát 109 PDF FIRE (2.704 trang) và 11 trang GT FIRE.

| Đo | Số |
|---|---|
| Hai vai trò ống cần tách: `main` (DN ≥ 65, thường thành mạch vòng hoặc trục cây) và `branch` (DN ≤ 50, rẽ tới đầu phun) | ngưỡng DN 65 trùng logic FIRE hiện có |
| Nhiều bản vẽ vẽ `main` và `branch` cùng màu đen, cùng độ dày (0,15 pt) | màu / độ dày đơn lẻ không đủ, cần topo và ngữ cảnh (chữ DN, đầu phun) |
| ~50 % dự án vẽ `main` bằng polyline có bề rộng → PDF thành vùng tô (`fs` / `f`, độ dày 0) | cần nhận dải tô hẹp |
| 2-core của đồ thị nối khớp `main` | F1 0,88–1,00 trên 9–10 / 11 trang có mạch vòng; sót trục chính trên trang thuần cây |
| `main` chiếm 20–40 % chiều dài ống nhưng < 5 % số shape | cần loss cân bằng theo chiều dài |

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Không có kiến trúc riêng cho từng hệ | dùng nguyên v46; đầu nét 64 → 3 (softmax), không đầu cạnh / đầu cụm khi dataset không có đối tượng | — | — |
| Mạch vòng / trục cây phân biệt `main` khi màu và độ dày như nhau | đặc trưng 2-core, độ sâu tỉa lá, dải tô, khoảng cách tới ký hiệu đầu cuối trong topo 18 chiều của v46; ảnh 2048 px đọc chữ DN | F2 `main` ≥ 0,85 | F2 `main` < 0,70 |
| `main` ít shape nhưng dài | CE trọng số `(chiều dài + 1)^0,5 × [1,0; 4,0; 2,5]` + Lovász-Softmax 0,7 + Lovász ống (1 ∪ 2) 0,3 | F2 `branch` ≥ 0,88 | F2 `branch` < 0,75 |
| Chuyển kiến thức CAD từ HVAC | khởi tạo từ checkpoint HVAC E151: giữ thân 17,3M, đầu nét 1 → 3 khởi tạo mới, bỏ đầu cụm / cạnh | F2 ống (1 ∪ 2) ≥ 0,92, không trang val nào < 0,70 | F2 ống < 0,85 hoặc có trang sập |

`best.pt` chọn theo `0,40·F2 ống + 0,60·trung bình(F2 main, F2 branch)`.

## 3. Chi tiết

- Dataset `fire_3cls_v1`: 46 trang đã rà bằng mắt, 19 dự án. Train 33 trang (12 dự án: 8 GT, 21 trang kho LAN đã rà, 4 trang âm toàn `other`).
  Val 13 trang (7 dự án giữ ngoài: 4 GT, 9 LAN). Không trùng dự án, SHA-256 hay vân tay đầu mút nét dương giữa train và val.
- Họng nước vách tường độc lập = `other`; tuyến kết hợp sprinkler + họng nước = `main`; drain / drencher / báo cháy / kích thước = `other`.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

E152, 25 epoch, val 13 trang, theo chiều dài.

| | **Kỳ vọng** | Đạt | Bác bỏ |
|---|---|---|---|
| F2 ống (1 ∪ 2) | ≥ 0,92 | ≥ 0,90 | < 0,85 |
| F2 `main` | ≥ 0,85 | ≥ 0,80 | < 0,70 |
| F2 `branch` | ≥ 0,88 | ≥ 0,85 | < 0,75 |
| trang val sập (F2 ống < 0,70) | 0 | 0 | ≥ 1 |

## 5. Lệnh

```bash
openstroke train data=/content/fire_3cls_v1 model=<best.pt của E151> lr0=0.001 warmup_epochs=2.0 paste=0.35 paste_mode=both \
    epochs=25 name=e152_os46_fire3_from_hvac
```

Train từ đầu: `model=openstroke47`. Đối chứng chỉ GT: thêm `folds=<fire_3cls_v1>/folds_gt11.json`.

Suy luận: `r.prob` = 1 − p(`other`), `r.cls_prob` (N, 3), `r.class_names` = `("other", "main", "branch")`.

## 6. Kết quả thật

**E152:** khởi tạo từ `last.pt` của E151, RTX PRO 6000. `best.pt` = epoch 5, dừng ở epoch 13 (từ epoch 6 Lovász train 0,132 → 0,064 trong
khi F2 val ở 0,5 giảm 0,853 → 0,811). 1:35 / epoch, 33–36 GB (E151 trước bản tối ưu: 4:16 / epoch, 60,7 GB).

| | Kỳ vọng | E152 (13 trang) | 4 trang GT | 9 trang LAN |
|---|---|---|---|---|
| F2 ống (1 ∪ 2) | ≥ 0,92 | 0,854 (P 0,907, R 0,842) | 0,791 | 0,864 |
| F2 `main` | ≥ 0,85 | 0,757 | 0,499 | 0,789 |
| F2 `branch` | ≥ 0,88 | 0,799 | 0,684 | 0,818 |
| trang sập | 0 | 3 | | |

- H176 **bị bác bỏ** vì có 3 trang val sập. F2 ống 0,854, `main` 0,757, `branch` 0,799 nằm giữa ngưỡng bác bỏ và mức đạt.
- Trên dự án lạ có kiểu vẽ đã gặp trong train, ba lớp đạt 0,94–0,98 (`uoa_q7 p018`: ống 0,978, main 0,972, branch 0,973).
- 3 trang sập là khổ A3 hoặc tổng mặt bằng thu nhỏ: 28 / 29 trang dương train là A1 / A0, augment tỉ lệ ±0,2 không phủ 0,4–0,6×.
- 76 % lỗi `main` → `branch` nằm trên 4 trang có `main` và `branch` cùng độ dày, không tô, không mạch vòng; gần như không trang train nào
  có tổ hợp này.
- Hướng E153: dữ liệu `fire_3cls_v2` (~110–150 trang dương, ưu tiên A3, tổng mặt bằng, mạng thuần cây cùng nét), augment tỉ lệ 0,45–1,5×
  và ép cùng độ dày, giữ nguyên 13 trang val.
