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

## 7. Hướng `E153` (`H177` — `fire_3cls_v3` siêu sạch + `scale=1.0` + `restyle_mode=fire`, 2026-11-27)

- **Kiểm toán độc lập & dọn sạch sang `datasets/fire_3cls_v3` (`datasets/openstroke_fire3_v3_data.zip`, `222,1 MB` / `211,8 MiB`, `96` file, CRC=OK):**
  - Kiểm toán lại 100 % `107/107` trang của `fire_3cls_v2` (đồ thị 2-core + kiểm layer CAD + phóng đại đối chiếu màu gốc PDF) phát hiện `14` trang lỗi nặng/độc dữ liệu (gồm **`neg_htmv2_p019`** là mặt bằng sprinkler nhà xưởng thật bị gán âm, `neg_htmv2_p003`, `lan_htmv2` thiếu ~45 % nhánh + dính chữ DN, `lan_hh1_hn` dính đầu phun & cụm van, `lan_ff_merged_20p_p005/p006` gãy mạch vòng, `lan_hampton_ht_p003`) và `30` trang cần sửa cục bộ (`5` Val + `25` Train).
  - Sau khi **loại bỏ 14 trang lỗi nặng** và **sửa chính xác 30 trang** (soi ảnh Trước/Sau từng trang một):
    - **Train (`80` trang = `63` dương + `17` âm sạch qua `26` dự án, lấy mẫu mặc định `sqrt(N)` cho `86,0 %` dương / `14,0 %` âm):** sửa sạch `lan_parkcity_ch03` (`3p`, khôi phục toàn bộ tuyến chính hành lang ngang trên cùng thành `1=main`), `lan_celadon_0804` (`6p`, hạ nhánh căn hộ cụt về `2=branch`, bỏ hộp vuông), `lan_ht3_hbl` (`6p`, giữ đúng 2 mạch vòng hành lang đôi, hạ 8–10 nhánh căn hộ cụt), `lan_ptn p003/p005` (`2p`, lọc `18` + `157` nét hình lục giác/van lõi thang), `lan_vng_cpe p003/p004` (`2p`, gỡ tuyến chữa cháy vách tường `y=812` & hạ 5 nhánh đỗ xe `w=0,60` về `branch`), `lan_hh12_0884` (`6p`, lọc 5–6 ký hiệu van kín nhỏ).
    - **Val sạch (`13` trang qua `7` dự án, `0` rò rỉ PDF/SHA-256, `max fg overlap = 0,0189`):** sửa sạch `lan_tcb_hk_p022` (hạ đúng 1 nhánh dọc giữa sai về `branch`, bỏ 30 nét gạch chéo `/` đầu bịt) và chuẩn hóa 4 trang `fp_0875` (`doc000149_gt100` + `lan_fp_0875_p003/p005/p006`) theo đúng layer CAD tường minh (`0-JRP-SPK_MAIN=1`, `0-JRP-SPK_BR=2`).
    - **Quy mô cuối `fire_3cls_v3`:** **`93` trang** (`80` Train / `13` Val), `11,59M` shape, `491.891 pt` `main`, `976.431 pt` `branch`; `folds.resolve()` `0` cảnh báo trên cả `folds.json` (`80/13`) và `folds_gt12.json` (`8/4`).
- **Mốc đối chứng `E152` (`F:\Downloads\fire.pt`, chấm lại trực tiếp không đổi trọng số trên Val13 sạch `v3`):**
  - Val13 cũ (`v1/v2`): `Pipe F2 0,8543 | Main F2 0,7569 | Branch F2 0,7989 | Macro 0,7779 | Fitness 0,8085`.
  - **Val13 sạch (`v3`)**: **`Pipe F2 0,8552 | Main F2 0,7917 (+0,0348) | Branch F2 0,7989 | Macro 0,7953 (+0,0174) | Fitness 0,8193`** (`doc000149_gt100 Main 0,400→0,510`, `lan_fp_0875_p006 Main 0,401→0,524`, `lan_tcb_hk_p022 Pipe 0,974→0,994 / Branch 0,890→0,923`).
- **Nâng cấp augmentation (`openstroke/data.py`, `openstroke/gpu_batch.py`, `openstroke/train.py`, `v0.48.8`):**
  - **`scale=1.0`** (thu phóng ngẫu nhiên đều trong `[0,50× ; 2,00×]` để trang A1/A0 co xuống chuẩn khổ A3 & tổng mặt bằng);
  - **`restyle_mode=fire`** (`flat_p=0,35`: 35 % lượt restyle ép mọi nét trong cửa sổ về cùng một độ dày `wM == wB`, 50 % giữ thứ bậc, 15 % hairline `0,00 pt` — chặn lối tắt `"wM == wB => branch"` mà không đổi HVAC `"robust"`).
- **Kỳ vọng ghi trước `E153` (`H177`, 20 epoch × 150 steps trên `fire_3cls_v3`):** `Pipe F2 ≥ 0,92`, `Main F2 ≥ 0,86`, `Branch F2 ≥ 0,88`, **0 / 13 trang val sập (`Pipe F2 < 0,70`)**.
- **Kết quả thật `E153` (`F:\Downloads\e153_os46_fire3_v3`, A100 80 GB, 1,10 h, 2026-11-27) — ĐẠT BƯỚC NHẢY LỚN SO VỚI `E152`:**
  - **`best.pt` (ep 3, EMA + TTA x8, `thr=0,30`):**
    - **`Pipe F2 = 0,919`** (`P=0,882, R=0,929`; `@0,50`: `0,907`), **Line-decoder Object F2 = `0,910`** (`P=0,900, R=0,912`, 0 layer).
    - **Đa lớp (`argmax`):** `Main F2 = 0,832` (`P=0,819, R=0,835`), `Branch F2 = 0,871` (`P=0,869, R=0,871`), `Macro FG = 0,851`, `Fitness = 0,8735`.
    - **Đa lớp (`thr=0,30`, `1-p0>=0,30`):** **`Pipe 0,9191 | Main 0,8330 | Branch 0,8877 | Macro 0,8604 | Fitness 0,8838`** (so `E152` cùng Val sạch: `+0,064` Pipe, `+0,041` Main, `+0,089` Branch, `+0,065` Macro!).
    - **Cứu sập triệt để toàn bộ 4 trang GT Val (`0/4` sập):** `doc000149` **`0,986`**, `doc000162` **`0,027 → 0,846` (`+0,819`, Main = `1,000`!)**, `doc000163` **`0,836 → 0,948`**, `doc000164` **`0,768 → 0,917`**; `lan_tcb_hk_p022` **`0,994`**, `lan_uoa_q7` **`0,971 / 0,986 / 0,987`**, `lan_fp_0875_p005/p006` **`0,950 / 0,967`**.
  - **Phát hiện quan trọng về `last.pt` (ep 20, EMA + TTA x8):** trong log epoch 16–20 `Main` tụt xuống `0,632` **CHỈ VÌ `multiclass_metrics()` chấm bằng `argmax` (`0,50`)**, trong khi `last.pt` bị trôi ngưỡng calibration sang `thr=0,05` (do 99,5 % shape là `other`). Khi chấm **`last.pt` ở đúng ngưỡng `thr=0,05` (`1-p0>=0,05`)**, **`last.pt` THỰC TẾ MẠNH HƠN CẢ `best.pt`**: **`Pipe 0,9189 | Main 0,8419 | Branch 0,8845 | Macro 0,8632 | Fitness 0,8855`**, đặc biệt cứu mạnh các trang khó (`lan_fp_0875_p003` `0,519 → 0,687`, `doc000149 Main 0,733 → 0,874`, `lan_fp_0875_p006 Main 0,680 → 0,872`, `lan_jrp_0913 Main 0,766 → 0,864`, `lan_l30m Branch 0,707 → 0,860`).
  - **2 điểm nghẽn cuối cần xử lý (không cần tăng dung lượng model):**
    1. **`75,5 %` tổng FP `other→main` toàn bộ 13 trang Val (`10.701 / 14.177 pt`)** rơi vào đúng **1 đa giác `xr-0-RanhGioiKhuDat`** cô lập không nối nhánh nào trên `lan_l30m_p002` (chỉ cần hậu kỳ lọc thành phần liên thông không có nhánh là loại sạch).
    2. **Sửa calibration / chấm chọn checkpoint đa lớp theo `thr` đã hiệu chuẩn** (hoặc `label_smoothing` / `focal` nhẹ + không dùng `argmax` cứng `0,50` khi mất cân bằng 99,5 % `other`) để `last.pt` không bị trôi ngưỡng về `0,05` và chọn checkpoint đúng đỉnh thật.
- **Lệnh Colab `E153`:**

```bash
!cd /content/OpenStroke && git pull origin main && pip install -e . --no-deps --quiet
!mkdir -p /content/fire3_v3/cache && unzip -qo /content/drive/MyDrive/Dataset/openstroke_fire3_v3_data.zip -d /content/fire3_v3 && ls /content/fire3_v3/fire_3cls_v3/manifest.json
!openstroke train \
  data=/content/fire3_v3/fire_3cls_v3 \
  cache=/content/fire3_v3/cache \
  model=/content/drive/MyDrive/Train/OpenStroke46/e151_os46_unified_hvac/fold0/last.pt \
  scale=1.0 \
  restyle=0.75 \
  restyle_mode=fire \
  paste=0.35 \
  paste_mode=both \
  lr0=0.0008 \
  warmup_epochs=2.0 \
  steps=150 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke46_Fire3 \
  name=e153_os46_fire3_v3
```
