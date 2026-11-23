# Train nét mảnh (FIRE / PLUMBING) — fire1, thin1

Ghi 2026-11-23 (sổ cái F187, F188, F213). Không phải version kiến trúc mới: cùng mạng E114 (openstroke28, chỉ bắt nét, không layer,
nhánh ảnh FPN), chỉ đổi dữ liệu. Kết quả thật điền sau khi có log Colab.

## Dữ liệu

| Dataset | Trang | Train | Val | Zip |
|---|---|---|---|---|
| `fire1` | 11 GT FIRE (AI_Takeoff trai_nc / FIRE) | 7 | 4 (doc149, doc162, doc163, doc164) | `datasets/openstroke_fire1_data.zip` 30 MB |
| `thin1` | 11 GT FIRE + 104 GT PLUMBING + 250 Viewer PLUMBING #38 | 7 + 96 + 190 | 4 + 8 + 60 | `datasets/openstroke_thin1_data.zip` 667 MB |

Nhãn: quy tắc v3 + E55 (nét dọc `Line` / `Ống …` của GT). `oinst` = object ống GT của nét (loss closure / cạnh dùng được), `ocls` = 0,
`fcls` = 1 trên nét ống. Nhãn Viewer phủ 75,8 % ống GT trên trang trùng (F188): thô, ít bắt nhầm. PLUMBING là một họ dự án CTN, nên
tổng quát trên bản vẽ lạ chỉ đo được ở 4 trang val FIRE.

## Giao thức đi kèm dataset (F213)

`manifest.json` có khoá `protocol`; trainer đọc tự động (`train.apply_protocol`), lệnh không cần cờ thêm.

- **Nhóm (cohort)** của trang = hệ + nguồn nhãn: `fire:gt`, `plumbing:gt`, `plumbing:viewer`.
- **Lấy mẫu cửa sổ train:** thin1 `fire:gt` 40 %, `plumbing:gt` 30 %, `plumbing:viewer` 30 % (trong mỗi nhóm vẫn theo √số nét). fire1 một nhóm.
- **Ngưỡng và best.pt:** chọn theo trung bình F2 nét của các nhóm `select` (thin1: `fire:gt`, `plumbing:gt`, mỗi nhóm nặng như nhau;
  Viewer chỉ báo, không dùng để chọn). fire1: `fire:gt`.
- **Không thừa hưởng từ checkpoint** các tuỳ chọn mô tả dữ liệu: E114 lưu `modes=hvac`, `viewer_share=0.65`; nếu thừa hưởng thì mọi trang
  FIRE / PLUMBING bị lọc. Cờ trên lệnh (`modes=`, `plumbing_share=`, `viewer_share=`) vẫn thắng.
- `gt_only=true` (split HVAC hvac10b) không dùng với hai dataset này; trainer báo lỗi nếu bật.

Log mỗi epoch in thêm: dòng `select` (thước đo chọn best), một dòng F2 nét cho từng nhóm (`*` = nhóm được chọn), F2 object theo giao thức
nét (nét ≥ ngưỡng nối bằng cạnh cùng object nếu mạng có đầu cạnh; E114 không có đầu cạnh nên mỗi nét là một object, số này thấp và chỉ
để so cùng mạng). Trước epoch 1 in **Before fine-tune** = E114 nguyên bản trên cùng val (đối chứng epoch 0). Cuối run có
`SUMMARY select (mean of …)` và `SUMMARY fire:gt / plumbing:gt / plumbing:viewer`.

## Colab

Tải hai zip lên Drive (ví dụ `MyDrive/Dataset/OpenStroke/`). Cài code:

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

Giải nén:

```python
!unzip -q -o /content/drive/MyDrive/Dataset/OpenStroke/openstroke_fire1_data.zip -d /content/thin
!unzip -q -o /content/drive/MyDrive/Dataset/OpenStroke/openstroke_thin1_data.zip -d /content/thin
```

EF1, chỉ FIRE:

```python
!openstroke train \
    data=/content/thin/fire1 \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStrokeThin \
    name=ef1_fire1
```

EF2, FIRE + PLUMBING:

```python
!openstroke train \
    data=/content/thin/thin1 \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStrokeThin \
    name=ef2_thin1
```

Công thức train lấy từ E114 (lr0 0,002, warmup 3 epoch, 200 bước × batch 4, cửa sổ 65.536, restyle `order`, `feature_drop=layer_stats`,
TTA ×8 ở val cuối). Hai run giống hệt nhau trừ dữ liệu. Gửi lại log là đủ.

## So sánh ghi trước (trước GPU)

- Đối chứng epoch 0: E114 (chỉ học HVAC) trên val FIRE.
- Thước đo chính: F2 nét `fire:gt` của mỗi run ở best.pt và theo từng epoch (cùng epoch so cùng epoch, vì best.pt của EF2 còn chọn theo
  PLUMBING). Phụ: `plumbing:gt`, `plumbing:viewer`, F2 object theo giao thức nét.
- Kỳ vọng (không phải kết quả): E114 trước fine-tune ≤ 0,60 F2 nét trên val FIRE; EF1 sau 10 epoch 0,65–0,80; EF2 ≥ EF1 + 0,02 nếu PLUMBING
  chuyển được sang FIRE.
- Tiêu chí: thin1 được dùng cho FIRE khi F2 `fire:gt` ở best.pt của EF2 ≥ EF1 + 0,02 **và** EF2 không thấp hơn EF1 ở ít nhất một nửa số
  epoch. Nhiễu seed lịch sử ~0,0125 (F33); 4 trang val nên chênh < 0,02 là trong nhiễu. Không đổi tiêu chí sau khi có số.

## Kết quả thật

Chưa chạy.
