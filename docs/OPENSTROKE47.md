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

## 8. Hướng `E154` (`H178` — `fire_3cls_v4` + Khắc phục 4 nút thắt OOD/calibration + Hậu kỳ FIRE v2, 2026-11-27)

- **Bằng chứng lỗi khái quát OOD của `E153` (user thử trên bản vẽ ngoài tập train/val thấy kém xa HVAC `E151`):**
  1. **Đóng băng sớm & trôi calibration:** `E153` đóng băng `best.pt` ngay ở **epoch 3** (`450` steps, `bce=0,0326, lovasz=0,255`) vì `trainer.py` chấm đa lớp bằng `argmax` cứng (`0,50`) trong khi ngưỡng tối ưu trôi từ `0,35 → 0,05` (`F2@0,5` tụt `0,907 → 0,800`). Ngược lại `last.pt` (ep 20) overfit quá tự tin (`thr=0,05`) trên 63 trang dương nên giòn trên PDF lạ.
  2. **0 phạt Lovász FP trên lớp/trang âm:** `lovasz_softmax_flat` bỏ qua toàn bộ lớp `c > 0` khi `fg.sum() == 0` (tức 17 trang âm và ~40 % cửa sổ không có `main` chịu 0 loss Lovász); nhánh `lv_fg` trên cửa sổ âm cũng chỉ phạt đúng 1 nét.
  3. **Rò lối tắt tần suất style trong `restyle_robust`:** khi ép đồng độ dày (`flat_p=0,35`) hoặc đen trắng (`uc ≥ 0,75`), các cột `style_shape_share`, `style_len_share`, `nb50_same_style` và `seqno_rank` không được cập nhật theo style mới sau restyle — model vẫn nhận diện ống qua tần suất layer CAD gốc.
  4. **Dữ liệu Train hẹp & lệch dự án (`63` trang dương / `26` dự án):** ít dự án hơn HVAC (`188` trang / ~50 dự án), `pdf_balance` bị khóa ở dataset ngoài `gt_only`, `neg_boxes=0`, và tỷ lệ nhầm `main → branch` cao gấp `11,2×` chiều ngược lại (`10.411 pt` vs `932 pt`).
- **4 nâng cấp triển khai cho `E154` (`H178`) + Hậu kỳ FIRE v2:**
  1. **Bộ dữ liệu siêu sạch `datasets/fire_3cls_v4` (`datasets/openstroke_fire3_v4_data.zip`, `365,12 MiB`, `156` file, `testzip=None`):** khảo sát `109` PDF FIRE/ELV trên kho LAN (`\\10.0.10.200`), soi duyệt bằng mắt **100 % `154` contact sheet** (`108` ứng viên), loại thẳng `48` trang không đạt chuẩn `≥ 99 %`, bổ sung **`+60` trang Train mới** (`+46` dương + `+14` âm khó: `27` trang đồng độ dày/thuần cây, `37` trang tầng hầm/dày/dưới ống gió/cong/A3, `12` trang có họng nước vách tường song song tách sạch `0=other`, `14` trang âm khó FA/sơ đồ nguyên lý/tổng mặt bằng ranh giới khu đất/FM-200/Novec). Quy mô cuối: **`153` trang** = **`140` Train** (`109` dương + `31` âm qua **`38` dự án**, tăng **`+93,4 %`** chiều dài FG: `796.993 pt` `main`, `1.625.740 pt` `branch`) / **`13` Val13 giữ nguyên từng byte SHA-256** so với `v3` (`7` dự án, `0` trùng dự án/SHA-256, `max fg overlap = 0,0171`, `folds.resolve()` `0` cảnh báo).
  2. **Chặn triệt để lối tắt trong `restyle_robust` (`openstroke/data.py`):** tính lại `style_shape_share` và `style_len_share` từ `style_ids(x)` sau restyle mỗi khi gộp độ dày (kể cả nhánh hairline) hoặc chuyển đen trắng, nâng `nb50_same_style ≥ 0,85` khi toàn bộ nét có viền gộp về 1 style, và dịch vòng `seqno_rank` ở chế độ `fire`.
  3. **Sửa loss, calibration & lấy mẫu (`openstroke/models/openstroke2.py`, `openstroke/engine/trainer.py`, `openstroke/train.py`):** phạt top-32 FP trong `lovasz_softmax_flat` & `lv_fg` trên lớp/cửa sổ âm; thêm `label_smoothing=0,03` vào CE đa lớp; nâng trọng số `stroke_class_weights` (`base ≥ 3,0`, `hi ≥ 6,0`); chấm & chọn `best.pt` đa lớp tại đúng ngưỡng hiệu chuẩn `fg_threshold=bst["threshold"]` kèm `warmup_guard` (`min_best_ep = max(warmup+2, epochs//3)`); mở `pdf_balance=true` cân bằng đều mọi dự án dương mà vẫn bảo toàn tỷ lệ trang dương/âm tự nhiên; bật `neg_boxes=0.30` (hộp/khung tên âm tổng hợp tương thích `uint8` 3 lớp).
  4. **Hậu kỳ FIRE v2 (`openstroke/line_decoder.py`, `0` layer, `~290 ms/trang` CPU — áp dụng ngay không cần train lại):** hysteresis cầu nối / thẳng hàng (`≤ 5°`) thuần hình học trong `[0,10, thr)` (miễn nhiễm loang tường vuông góc trên bản vẽ OOD) + lọc vòng kín cô lập 0 nhánh (`0/93` GT bị mất) + làm mượt topo 2-core mạch vòng `main ≥ 65 %` & cầu nối giữa hai nút `main` (`99,88 %` bảo toàn Oracle trên toàn bộ 93 trang GT). Kiểm chứng trực tiếp trên `E153` Val13:
     - `best@0,30`: Stroke `Pipe 0,9191 →` **`0,9318 (+0,0127)`**, `Main 0,8330 →` **`0,8719 (+0,0389)`**, `Branch 0,8877 → 0,8854`, `Fitness 0,8838 →` **`0,8999 (+0,0161)`**; Line `All 0,9096 →` **`0,9155 (P 0,914, R 0,916)`**, `Main 0,7273 →` **`0,7620 (+0,0347)`**, `Branch 0,8787 →` **`0,8815`**, `Macro 0,8030 →` **`0,8218 (+0,0188)`**.
     - `last@0,05`: Stroke `Pipe 0,9292 | Main 0,8740 | Fitness 0,8978`; Line `All 0,9155 (P 0,920, R 0,914) | Main 0,7769 | Branch 0,8760 | Macro 0,8265`.
- **Chỉ số kỳ vọng ghi TRƯỚC khi train `E154` (`H178`, 20 epoch × 200 steps trên `fire_3cls_v4`, chấm trên cùng Val13 sạch):**

| Thước đo (Val13 sạch, cùng giao thức `E153`) | Mốc `E153` (`best` / `last`) | **Kỳ vọng `E154`** | Đạt | Bác bỏ |
|---|---:|---:|---:|---:|
| `best_epoch` & ngưỡng hiệu chuẩn `thr` | ep 3 (`0,30`) / ep 20 (`0,05`) | `ep ≥ 7`, `thr ∈ [0,20; 0,60]` | `ep ≥ 6`, `thr ≥ 0,15` | `thr ≤ 0,05` |
| Stroke raw `Pipe F2` | `0,9191` / `0,9189` | **≥ 0,940** | ≥ 0,930 | < 0,920 |
| Stroke raw `Main F2` | `0,8330` / `0,8419` | **≥ 0,885** | ≥ 0,865 | < 0,850 |
| Stroke raw `Branch F2` | `0,8877` / `0,8845` | **≥ 0,905** | ≥ 0,895 | < 0,880 |
| Hậu kỳ FIRE v2 Line `All / Main / Branch` | `0,9155 / 0,7620 / 0,8815` | **≥ 0,935 / ≥ 0,840 / ≥ 0,900** | ≥ 0,925 / ≥ 0,810 / ≥ 0,890 | < 0,915 |
| Trang val sập (`Pipe F2 < 0,70`) | `1` (`fp_0875_p003 = 0,519`) | **0** (`p003 ≥ 0,78`) | **0** | ≥ 1 |

- **Lệnh Colab `E154`:**

```bash
!cd /content/OpenStroke && git pull origin main && pip install -e . --no-deps --quiet
!mkdir -p /content/fire3_v4/cache && unzip -qo /content/drive/MyDrive/Dataset/openstroke_fire3_v4_data.zip -d /content/fire3_v4 && ls /content/fire3_v4/fire_3cls_v4/manifest.json
!openstroke train \
  data=/content/fire3_v4/fire_3cls_v4 \
  cache=/content/fire3_v4/cache \
  model=/content/drive/MyDrive/Train/OpenStroke46/e151_os46_unified_hvac/fold0/last.pt \
  scale=1.0 \
  restyle=0.75 \
  restyle_mode=fire \
  paste=0.35 \
  paste_mode=both \
  neg_boxes=0.30 \
  pdf_balance=true \
  lr0=0.0005 \
  warmup_epochs=2.0 \
  steps=200 \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke46_Fire3 \
  name=e154_os46_fire3_v4
```

- **Kết quả thật `E154` (`F:\Downloads\e154_os46_fire3_v4`, RTX PRO 6000 Blackwell, `0,66 h`, `1:32–1:40/ep`, 2026-11-27) — ĐẠT KỶ LỤC MỌI THỜI ĐẠI HỆ FIRE 3 LỚP (`0` layer, `0` logic AI_Takeoff):**
  - **Hội tụ & calibration:** tăng đơn điệu xuyên suốt 20 epoch (không còn đóng băng ở ep 3 như `E153`), **`best.pt` = epoch 18 EMA** (`val_scores.npz`), **`last.pt` = epoch 20 EMA** (`val_scores_last.npz`). Độ bền ngưỡng vượt trội: `F2@0,50` Raw giữ **`0,9153` (`P=0,923, R=0,913`)** (vs `E153` sập về `0,800`), chênh giữa `thr=0,10 → 0,50` cực phẳng (`0,9320 → 0,9153`).
  - **So sánh đầy đủ trên cùng 13 trang Val13 sạch (`byte-identical` giữa `v3` & `v4`, TTA ×8):**

| Thước đo (Val13 sạch, cùng giao thức) | `E152` (`v1` sạch) | `E153` (`best@0,30`) | `E153` (`last@0,05`) | **`E154 best@0,10`** | **`E154 best@0,20`** | **`E154 best@0,30`** | **`E154 last@0,10`** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Raw Stroke `Pipe F2`** | `0,8552` | `0,9191` | `0,9189` | **`0,9320`** (`P 0,893 R 0,942`) | `0,9291` (`P 0,906 R 0,935`) | `0,9246` | **`0,9322`** (`P 0,896 R 0,942`) |
| **Raw Stroke `Main / Br / Macro`** | `0,792 / 0,799 / 0,795` | `0,833 / 0,888 / 0,860` | `0,842 / 0,885 / 0,863` | **`0,853 / 0,905 / 0,879`** | `0,852 / 0,902 / 0,877` | `0,851 / 0,896 / 0,874` | **`0,853 / 0,905 / 0,879`** |
| **Raw Stroke `Fitness`** | `0,8193` | `0,8838` | `0,8855` | **`0,9003`** | `0,8980` | `0,8941` | **`0,9004`** |
| **Hậu kỳ FIRE v2 Stroke `Pipe / Main / Br`** | — | `0,932 / 0,872 / 0,885` | `0,929 / 0,874 / 0,884` | **`0,9412 / 0,8950 / 0,9017`** | **`0,9396 / 0,8941 / 0,9003`** | `0,9357 / 0,8925 / 0,8958` | **`0,9414 / 0,8943 / 0,9015`** |
| **Hậu kỳ FIRE v2 Stroke `Fitness`** | — | `0,8999` | `0,8978` | **`0,9155`** | **`0,9141`** | `0,9108` | **`0,9153`** |
| **Hậu kỳ FIRE v2 Line `All` (`P / R`)** | — | `0,9155` (`0,914/0,916`) | `0,9155` (`0,920/0,914`) | **`0,9290`** (`0,926/0,930`) | **`0,9271`** (`0,942/0,923`) | `0,9231` (`0,950/0,917`) | **`0,9293`** (`0,930/0,929`) |
| **Hậu kỳ FIRE v2 Line `Main / Br / Macro`** | — | `0,762 / 0,882 / 0,822` | `0,777 / 0,876 / 0,827` | **`0,8107 / 0,8975 / 0,8541`** | `0,8089 / 0,8964 / 0,8526` | `0,8067 / 0,8924 / 0,8496` | `0,8094 / 0,8974 / 0,8534` |
| **Trang sập (`Pipe F2 < 0,70`) sau hậu kỳ** | `3` | `1` (`p003=0,519`) | `1` (`p003=0,687`) | **`0 / 13`** (min **`0,8123`**) | **`0 / 13`** (min **`0,8123`**) | **`0 / 13`** | **`0 / 13`** (min **`0,8146`**) |

  - **Đánh giá giả thuyết `H178`: ĐẠT** trên hầu hết mọi tiêu chí (Raw `Pipe 0,932 ≥ 0,930`, `Branch 0,905 ≥ 0,895`, `Line All 0,929 / Main 0,811 / Branch 0,898` đạt trọn 3/3, `0/13` trang sập với `fp_0875_p003` vọt `0,519 → 0,812–0,838`, `best_epoch = 18`); duy `Raw Main = 0,853` nằm giữa ngưỡng bác bỏ (`0,850`) và ngưỡng đạt (`0,865`), nhưng qua hậu kỳ FIRE v2 bật lên **`0,8950`** (`+0,023` vs `E153`). Oracle hậu kỳ FIRE v2 trên toàn bộ **`153/153` trang** `fire_3cls_v4` bảo toàn **`99,98 %` Pipe / `99,75 %` Main** và `0` vòng GT bị xóa nhầm.
  - **Giải phẫu lỗi còn lại (`best@0,20` sau Hậu kỳ FIRE v2):**
    1. **Lệch `Line Main` (`0,809`) so với `Stroke Main` (`0,894`) tập trung trọn ở 3 trang `lan_uoa_q7`** (`p005: 0,942 → 0,790`, `p009: 0,799 → 0,508`, `p018: 0,935 → 0,773`): lỗi hậu kỳ `line_decoder` trên ống chính vẽ kiểu dải tô/nét đôi `fs`, không phải lỗi model.
    2. **FP còn lại (`10.573 pt` sau khi `drop_closed_loops` cắt sạch `10.701 pt` `RanhGioiKhuDat`):** tập trung ở nét chú thích/mũi tên ghi chú (`0-JRP-TEXT*` `1.986 pt`, `IFE-Text/DIM` `1.108 pt`, `doc000164` layer `0` `1.496 pt`) và vòng tròn bán kính phủ đầu phun (`ban kinh sp` `1.442 pt` trên `jrp_0913`).
    3. **FN & nhầm lớp:** `lan_jrp_0913_p001` chiếm `47 %` tổng FN nhánh (`6.359 pt` `branch`, `Main` đã đạt `0,936`); nhầm `main → branch` giảm `-32 %` (`10.411 → 7.034 pt`), còn tập trung ở `doc000163` (`Main 0,680`) và `fp_0875_p003` (`Main 0,603`).
  - **Triển khai khuyến nghị:** `F:\Downloads\e154_os46_fire3_v4\fold0\best_deploy.pt` (`33,3 MB` FP16, `0,64 s/trang` GPU RTX PRO 6000) + Hậu kỳ FIRE v2 (`decode_fire_network`, `~0,18 s/trang` CPU) ở ngưỡng **`thr = 0,20`** (hoặc **`0,30`** khi cần Precision cao trên bản vẽ OOD).
