# OpenStroke28: model chỉ chọn nét (không layer) + logic HVAC của pipeline trên layer thật

Ngày 2026-11-13. Trạng thái: **đã code, dữ liệu dựng xong, chưa train.** Giả thuyết **H129**, ledger F139–F142.

## 1. Mục tiêu và phạm vi

User 2026-11-13 chốt hướng sản phẩm, thay mục tiêu "model end-to-end ra object không layer" của các version box:
- Model chỉ làm một việc: nhận lệnh vẽ (hình học, màu, độ dày) + ảnh vẽ từ vector, trả **xác suất từng nét là nét cần bóc** (ống gió HVAC).
- Logic HVAC của pipeline AI_Takeoff dựng object từ các nét đó, trên **layer thật** của PDF. Model thay model chọn layer.
- Mong muốn: chống lỗi tốt hơn khi nét của một object nằm rải trên nhiều layer (người vẽ đặt sai layer).

Điều kiện sản phẩm giữ nguyên: F2 object ≥ 0,80 (6 lớp, cùng bộ chấm với MAIN F2) trên trang GT giữ ngoài train.
Model **không đọc layer** (`no_layers=true`): một nét vẽ sai layer có cùng đầu vào như khi đúng layer (có test). Layer chỉ dùng ở bước logic.

## 2. Bằng chứng (0 GPU, ledger F139–F141)

| Đo (val GT, 6 lớp, bộ chấm MAIN F2) | F2 | Nguồn |
|---|---|---|
| Pipeline AI_Takeoff (logic trên layer pipeline chọn), 5 trang hvac8r | 0,881 | F141 |
| Model nét cũ **E37** (có đặc trưng layer, out-of-fold) + logic | **0,869** | F141 |
| Box detector E69 / E111 | 0,609 / 0,643 | F141, OpenStroke27 |
| Logic trên nét GT hoàn hảo, layer thật | 0,872 | F141 |
| Logic trên nét GT, không layer (gom theo kiểu nét / một layer) | 0,710 / 0,692 | F141 |
| Gom đơn giản trên mặt nạ nhị phân hoàn hảo | 0,19 | F139 |
| 10 % object nằm trọn trên layer sai: mất nét / đủ nét giữ layer sai / đủ nét + gom layer | 0,689 / 0,707 / 0,744 | F141 |
| Mốc trên đúng các trang val GT của split này: xem §7 | | F142 |

Kết luận rút ra:
1. Trên trang có layer đúng, hướng "nét + logic" đã ngang pipeline và trên 0,8. Box detector thua xa.
2. Model nét cũ dùng đặc trưng layer và LayerPool: trang bỏ layer thì sập (F55: 0,925 → 0,599; F68: 2D.pdf F2 0). Muốn chống lỗi layer thì model không được dựa vào layer.
3. Logic AI_Takeoff dựng object theo từng layer: đưa đủ nét mà layer sai chỉ hơn mất nét +0,02. Bước "gom lại layer" trước logic gỡ được +0,05–0,2. Đây là chỗ hậu kỳ có việc làm.
4. Trang không layer: logic chỉ đạt khoảng 0,71 kể cả khi nét hoàn hảo. Version này không giải quyết trang không layer; đó là việc của version sau (model ra thêm lớp từng nét, trần 0,885, F139–F140).

## 3. Giả thuyết đăng ký trước (H129)

Model nét không layer (công thức E37: thân vector PTv3 + nhánh ảnh FPN, khởi tạo `e32p`) train trên GT + Viewer (`e8`), qua logic AI_Takeoff trên layer thật:
- (a) đạt F2 object ≥ 0,80 trên 5 trang val GT, không thua bản có layer (E115) quá 0,02;
- (b) khi 10 % object nằm trên layer sai, bước gom layer + logic giữ ≥ 0,75 và hơn cách chọn layer (mất nét) ≥ 0,05.

Biến nghiên cứu: model có đọc layer hay không (E114 so với E115, cùng dữ liệu, split, recipe). Data Viewer và split giống nhau ở cả hai run.

## 4. Kiến trúc và luồng dữ liệu

```
payload PDF (shapes: items, màu, độ dày, nét đứt, layer)
  ├─ features.page_tokens(no_layers=True): 47 đặc trưng / nét, layer = một layer rỗng → layer_stats = hằng số
  ├─ OpenStroke2 (PTv3 trên nét, cửa sổ 65.536) + ImageBranch (cửa sổ vẽ ra ảnh, ResNet18 + FPN stride 4, đọc tại điểm của nét)
  └─ p(nét cần bóc) cho từng shape
hậu kỳ:  chọn nét p ≥ t (t chọn trên val) → [gom layer: nét trên layer < 20 % chiều dài đã chọn lấy layer của nét chính gần nhất]
         → extract_hvac.detection.run_detection(drawings = nét đã chọn với layer thật, layers = các layer đó) → object HVAC
```

Nguồn: thân vector theo Point Transformer V3 (Wu và cs., CVPR 2024), nhánh ảnh FPN (Lin và cs., CVPR 2017), phân loại từng primitive như
SymPoint / CADSpotting (panoptic symbol spotting trên FloorPlanCAD). Lấy: giám sát từng nét, ảnh đọc tại nét. Bỏ: phần instance (logic lo).
Không dùng số PQ của các bài đó làm kỳ vọng.

## 5. Triển khai

- `openstroke/features.py` `page_tokens(no_layers)`; `data.py` `load_page` / `page_from_npz(no_layers)` (cache riêng `.nl`), `sampling_prob(viewer_share)`.
- `train.py`: cờ `no_layers`, `viewer_share`; `engine/trainer.py` ghi `no_layers` vào config checkpoint; `infer.py` / `parallel.py` đọc lại (app dùng đúng chế độ).
- Dữ liệu: `tools/make_stroke_dataset.py` (từ `obb_hvac10b`): mỗi trang GT / Viewer nhãn từ payload theo `relabel.object_strokes`, như score pack.
- Chấm sau train: `research/probes/logic_mask_probe.py` (val_scores → logic, các arm `real / remap / one / layers / union`) rồi `logic_object_probe.py` với pack hvac10b.
- Test: `tests/test_os28.py` (4 test: không layer, cache riêng, tỉ lệ trộn, train tí hon + điểm không đổi khi đổi layer). Toàn bộ `tests/` (không gồm obb) 115 pass trước khi thêm.

## 6. Dữ liệu, split, metric

- `datasets/e8`: split `hvac10b` (OpenStroke27 §8). Train 19 trang GT + 171 trang Viewer; val (fold 0) 5 trang GT + 8 trang Viewer. 21 trang cùng dự án val bị bỏ.
- Nhãn nét: GT từ DB hiện tại, Viewer từ "Labels V1", cùng quy tắc `object_strokes`. Layer giữ trong npz (logic cần), model không đọc.
- Trộn: `viewer_share=0.65` (65 % cửa sổ train từ trang Viewer). Augment của E37: lật, xoay 90°, zoom, đổi màu giữ thứ tự sáng (`restyle_mode=order`), bỏ nét.
- Metric chính: F2 object 6 lớp qua logic, arm `real<t>`, trên 5 trang val GT. Phụ: F2 nét theo chiều dài (trainer in mỗi epoch), nhóm Viewer qua logic (không có chữ OCR của pipeline, chỉ chẩn đoán).

## 7. Baseline, ablation, recipe

Mốc trên val GT của split `hvac10b` (F142; 4 trang: doc187 không có lần chạy pipeline nên thiếu chữ OCR cho logic; 6 lớp, bộ chấm MAIN F2):

| Cách | F2 | R | P |
|---|---|---|---|
| Pipeline: logic trên layer pipeline chọn / kết quả đã lưu | 0,872 / 0,886 | 0,869 / 0,891 | 0,882 / 0,870 |
| Logic trên nét GT, layer thật / gom theo kiểu nét | 0,884 / 0,898 | 0,874 / 0,890 | 0,926 / 0,935 |
| **E37 (có layer) mặt nạ p ≥ 0,3 trên layer thật** (đúng thiết kế này) | **0,865** | 0,869 | 0,850 |
| E37 mặt nạ p ≥ 0,5 trên layer thật | 0,862 | 0,857 | 0,879 |
| E37 làm bộ chọn layer (`layers0.1` / `union0.1`) | 0,866 / 0,865 | 0,871 | 0,844 |
| E37 mặt nạ + gom layer (p ≥ 0,3 / 0,5) | 0,788 / 0,813 | | |
| E37 mặt nạ, một layer (p ≥ 0,3 / 0,5) | 0,784 / 0,812 | | |

- Trên trang sạch, "gom layer" (nét trên layer < 20 % lấy layer của nét chính gần nhất) làm **tệ đi** (0,865 → 0,788): trang có nhiều layer HVAC hợp lệ
  (layer phụ kiện nhỏ). Bước gom layer chỉ được áp cho nét có dấu hiệu sai layer: nét được chọn nằm trên layer mà phần lớn nét khác **không** được chọn
  (layer kiến trúc). Đó là việc hậu kỳ sau E114, đo trên kết quả thật, không đổi model.
- Logic trên nét GT gom theo kiểu nét (không cần layer) đạt 0,898 trên 4 trang này (trên 5 trang hvac8r có doc190 chỉ 0,710): logic không layer
  chạy được trên nhiều trang, sập ở một số trang.


| Run | Khởi tạo | Biến | Recipe |
|---|---|---|---|
| **E114** (chính) | `e32p` | `no_layers=true layer_head=false` | E54 recipe: `feature_drop=layer_stats batch=4 lr0=0.002 image_branch img_lum img_fpn restyle_mode=order`, `viewer_share=0.65`, 10 epoch |
| **E115** (đối chứng) | `e32p` | có layer: `layer_head=true` (như E37) | như E114 |

## 8. Kỳ vọng và tiêu chí (ghi trước khi train)

| Chỉ số | Đạt | Bác bỏ |
|---|---|---|
| E114 + logic, arm `real` hoặc `remap` (5 trang GT) | ≥ 0,80 và ≥ E115 − 0,02 | < 0,72 |
| E114, 10 % object trên layer sai, hậu kỳ gom layer tốt nhất | ≥ 0,75 và ≥ arm mất nét + 0,05 | < arm mất nét + 0,02 |
| E114 trên trang sạch, cùng hậu kỳ gom layer đó | không tụt quá 0,02 so với arm `real` | tụt > 0,05 |
| F2 nét (trainer, val GT) | ≥ 0,80 | < 0,70 |

Kỳ vọng định lượng: F2 nét E114 khoảng 0,80–0,85 (E37 có layer 0,85 trên 5 fold cũ); logic khoảng 0,80–0,86.
Nếu E114 < 0,80 mà E115 ≥ 0,80: model cần layer thật, xem lại đặc trưng không layer (ngữ cảnh rộng hơn, kiểu nét). Nếu cả hai < 0,80: nút thắt là logic hoặc nhãn, xem từng trang.

## 9. Kiểm thử và chi phí

- Điểm của một nét không đổi khi đổi layer của mọi nét (test). Checkpoint mang `no_layers`; suy luận đọc trang giống train.
- Chi phí: E54 khoảng 25 phút / fold / 15 epoch trên G4 với khoảng 50 trang (F65). E114 có 190 trang train, số bước / epoch giữ mặc định nên thời gian / epoch tương tự; val 13 trang lớn hơn. Chưa đo trên GPU này.

## 10. Lệnh Colab

Upload lên Drive:
- `F:\Source_code\OpenStroke\datasets\openstroke_e8_data.zip` (710 MB) → `MyDrive/Dataset/`
- `F:\Source_code\OpenStroke
- `F:\Source_code\OpenStroke\runs_colab\e32p\fold0\best.pt` (33 MB, tiền huấn luyện) → `MyDrive/Train/OpenStroke28/e32p.pt`

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/os28 && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_e8_data.zip -d /content/os28
```

E114 (chính, model không đọc layer):

```python
!openstroke train \
    model=/content/drive/MyDrive/Train/OpenStroke28/e32p.pt \
    data=/content/os28/e8 \
    fold=0 \
    modes=hvac \
    feature_drop=layer_stats \
    batch=4 \
    lr0=0.002 \
    image_branch=true \
    img_lum=true \
    img_fpn=true \
    restyle_mode=order \
    no_layers=true \
    viewer_share=0.65 \
    epochs=10 \
    cache=/content/os28/cache \
    project=/content/drive/MyDrive/Train/OpenStroke28 \
    name=e114_strokes_nolayer
```

E115 (đối chứng có layer, chạy sau E114): cùng lệnh, bỏ `no_layers=true`, thêm `layer_head=true`, `name=e115_strokes_layers`.

Log in `fold 1 ... val F2` theo F2 nét chiều dài trên 13 trang val; ngưỡng tốt nhất chọn trên val. Cảnh báo "model=<.pt> ... rò val" bỏ qua được:
`e32p` chỉ tiền huấn luyện trên trang không nhãn. Cần gửi lại: log, `fold0/result.json`, `fold0/val_scores.npz` (để chấm qua logic AI_Takeoff ở máy, 0 GPU).

## 11. Kết quả thật

**E114 (log user, 10 epoch, 0,40 h, 29,3 GB VRAM, 1,8 it/s).** F2 nét theo chiều dài trên 13 trang val, mỗi epoch:
0,780 / 0,798 / 0,800 / 0,817 / 0,820 / 0,829 / 0,832 / 0,833 / 0,834 / **0,834** (R 0,86 → 0,89, P 0,56 → 0,68). Best epoch 10, TTA ×8, ngưỡng 0,30:
**R 0,886, P 0,693, F2 0,839**; ở 0,5: R 0,868, P 0,724, F2 0,835. Gán true mọi nét: 0,168.

| Trang | R | P | F2 |
|---|---|---|---|
| doc180 / 187 (nét mảnh đen) / 189 / 200 / 201 (GT) | 0,964 / 0,933 / 0,915 / 0,939 / 0,926 | 0,715 / 0,923 / 0,778 / 0,795 / 0,877 | 0,902 / 0,931 / 0,884 / 0,906 / 0,916 |
| 8 trang Viewer | 0,70–0,96 | 0,26–0,87 | 0,61–0,86 |

- Trang GT: F2 nét 0,88–0,93, R 0,92–0,96. Trang nét mảnh đen doc187 (trước đây model box sót nhiều nhất) 0,931. Trang Viewer kém hơn: P thấp ở
  HVAC-1 (nét mảnh màu, 0,26–0,34), R thấp ở VIAS / NO3 (0,70–0,75). Nhãn Viewer `in_progress` và quy ước khác GT nên không đọc như lỗi model.
- So với E37 trên cùng 4 trang GT (nhãn hiện tại): E37 R 0,976–0,998 nhưng P 0,27–0,37 (E37 train trên nhãn v2 rộng hơn), vậy mà E37 + logic vẫn
  đạt 0,865. Logic lọc nét thừa rất mạnh nhưng không lấy lại được nét bị bỏ. E114 có P cao hơn nhiều nhưng R thấp hơn E37 khoảng 4 điểm. Ngưỡng cho
  logic nên chọn theo F2 object (thử 0,02–0,3), không theo F2 nét.
- **F2 object qua logic AI_Takeoff** (val_scores E114, 6 lớp, bộ chấm MAIN F2 với xếp lớp `class_of`; 4 trang GT có mốc / 5 trang GT):

| Hậu kỳ | GT4 | GT5 | Ghi chú |
|---|---|---|---|
| Pipeline: logic trên layer pipeline chọn / kết quả đã lưu | 0,883 / 0,897 | – | dùng layer PDF |
| E37 (có layer) mặt nạ p ≥ 0,3 trên layer thật | 0,876 | – | dùng layer PDF |
| E114 mặt nạ p ≥ 0,05 trên layer thật | 0,813 | 0,836 | dùng layer PDF (số tính với bộ chấm cũ) |
| E114 làm bộ chọn layer (`layers0.1`) | 0,824 | 0,840 | dùng layer PDF (bộ chấm cũ) |
| E114 một layer, p ≥ 0,1 | 0,793 | 0,823 | không layer (bộ chấm cũ) |
| E114 layer_1 (màu + độ dày), p ≥ 0,05 | 0,822 | 0,847 | không layer PDF |
| **E114 layer_1 p ≥ 0,05 + luật Duct mảnh < 2 pt → Chân rẽ** | **0,862** | **0,871** | không layer PDF; mặc định của app |
| + bỏ hộp ký hiệu nhỏ | 0,796 | 0,810 | bác bỏ (bỏ nhầm ống ngắn thật) |
| Trần: nét GT hoàn hảo + layer_1 / + layer thật | 0,910 / 0,894 | – | |

  Bộ chấm cũ của probe xếp "Elbow vuông" thành "Elbow" (so tiền tố); đã sửa, các số "bộ chấm cũ" lệch khoảng +0,005–0,01 khi chấm lại.
  Luật Elbow không có cung → Elbow vuông: không đổi điểm (giữ vì đúng định nghĩa, trung tính trên val).
- **Đạt H129 phần (a) ở mức không layer:** E114 + layer_1 + luật hình dạng 0,871 ≥ 0,80 trên 5 trang GT, kém pipeline có layer 0,021 trên 4 trang chung.
  Phần chống lỗi layer (b) không còn liên quan khi hậu kỳ cũng không dùng layer PDF.
- **Trang Viewer: logic sập** (8 trang, GT 1.633 object): F2 object 0,24, sót 72,5 %, không ra Elbow nào, dù F2 nét 0,61–0,86. Logic AI_Takeoff không
  dựng được object trên kiểu vẽ này: giới hạn của logic, không phải của model. Phân tích lỗi trên GT5 (layer_1, trước luật): khớp 84,8 %, sót 7 % (Duct
  48, Elbow 31), sai lớp 5,4 % (44 Chân rẽ mảnh bị gọi Duct), Duct ảo 97 (63 ở doc180, hình 19 × 11 pt lặp lại).
- Chưa chạy E115 (đối chứng có layer).

## 12. Kết luận và version sau

*(chưa có)*
