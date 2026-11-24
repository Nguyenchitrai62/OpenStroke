# Train nét mảnh (FIRE / PLUMBING) — fire1, thin2 (F227 sửa thin1)

Ghi 2026-11-23 (sổ cái F187, F188, F213) và cập nhật 2026-11-24 (F227: kiểm toán toàn bộ dữ liệu FIRE + PLUMBING, sửa 3 lỗi đồng bộ/lọc nhãn và build `thin2` thay thế `thin1`). Không phải version kiến trúc mới: cùng mạng E114 (openstroke28, chỉ bắt nét, không layer, nhánh ảnh FPN), kết hợp bộ giải mã vector trực tiếp `openstroke/line_decoder.py` (F225). Kết quả thật điền sau khi có log Colab.

## Dữ liệu

| Dataset | Trang | Train | Val | Zip |
|---|---|---|---|---|
| `fire1` | 11 GT FIRE (AI_Takeoff trai_nc / FIRE) | 7 | 4 (doc149, doc162, doc163, doc164) | `datasets/openstroke_fire1_data.zip` 30 MB |
| `thin1` *(cũ — lỗi, không dùng)* | 11 GT FIRE + 104 GT PLUMBING + 250 Viewer PLUMBING #38 | 7 + 96 + 190 | 4 + 8 + 60 | `datasets/openstroke_thin1_data.zip` 667 MB |
| **`thin2` *(F227 — bản sạch)*** | **11 GT FIRE + 103 GT PLUMBING + 290 Viewer PLUMBING #38** | **7 + 95 + 214** | **4 + 8 + 76** | **`datasets/openstroke_thin2_data.zip` 717 MB** |

Nhãn: quy tắc v3 + E55 + F227 (`_rescue_needed_stray` cứu lớp ống thật có `< 10 %` chiều dài trang khi dùng chung layer với sơ đồ không gian 3D). `oinst` = object ống GT của nét, `ocls` = 0, `fcls` = 1 trên nét ống.
**Kiểm toán F227 (2026-11-24):**
- `fire:gt` (11 trang) và `plumbing:gt` (103 trang sạch): chuẩn vàng (~98–99 %); phần không gán nhãn ở nửa dưới cùng trang PLUMBING là sơ đồ không gian 3D (chủ đích để `other`).
- `plumbing:viewer` (#38 `line_junction`): độ phủ thấp `75,8 %` ở `thin1` là do **lỗi parser `tools/make_thin_dataset.py` bỏ qua `line_edges`** (rơi 47,1 % cạnh ống nhánh, nối chéo 19.553 đường giả) + **2 khung `Mặt bằng` `draw_mode="obb"` gán nhầm lớp `line1`** + **lọc `stray < 10 %` xóa mất lớp ống `CPG-PS-CWP` / `Repair — Pen 0.84pt` trên 4 trang GT train, 3 trang Viewer val và loại oan 13 trang Viewer**. Sau khi sửa và build `thin2`: trên 110 cặp `viewer_vs_gt`, Recall có trọng số nhảy **`0,7577 → 0,9624`**, Precision **`0,9304 → 0,9735`**; trên 105 cặp đúng cùng bản vẽ (`shared ≥ 0,993`), **`103/105` (98,1 %) đạt cả R ≥ 0,95 và P ≥ 0,95** (median `1,000 / 1,000`, chỉ đúng 1 trang `vw_0bf4_p35` thực sự thiếu nhãn người vẽ `R = 0,478`). Chỉ loại duy nhất 1 bản vẽ (`doc000265_p3` / `vw_39da_p18`) bị CAD băm thành 12.500 đoạn `0,3 pt`. PLUMBING là một họ dự án CTN, nên tổng quát trên bản vẽ lạ vẫn đo chính ở 4 trang val FIRE.

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
!unzip -q -o /content/drive/MyDrive/Dataset/OpenStroke/openstroke_thin2_data.zip -d /content/thin
```

EF2 (bản sạch `thin2`, FIRE + PLUMBING, 1 model chung + `restyle_mode=robust` F228 chống lối tắt nét mảnh/màu đen):

```python
!openstroke train \
    data=/content/thin/thin2 \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    restyle_mode=robust \
    restyle=0.7 \
    epochs=15 \
    project=/content/drive/MyDrive/Train/OpenStrokeThin \
    name=ef2_thin2
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
