# OpenStroke38 — ảnh quyết định, vector chỉ chỉ chỗ; con mắt OpenOBB (H153–H154, E135–E136)

Ý của user 2026-11-23: người bóc tách nhìn thấy hình dạng giống ống, zoom vào, rồi đánh lên các nét vector; model cần mô phỏng chuỗi đó. Mục tiêu
sản phẩm không đổi: model nhả ra nét cần bóc / object, hậu kỳ nhẹ. Thẻ này là **bước 1**: kiểm xem nhánh vector có đang là nguồn học thuộc không,
khi để ảnh ra quyết định và vector chỉ cho biết nét nằm ở đâu. Bước 2 (zoom thật) chỉ làm nếu bước 1 đạt.

## 1. Bằng chứng từ các version trước

- Học thuộc ở mức nét: trang train 0,993, trang GT lạ 0,933 (F182); lệch nằm ở P (bắt thừa nét trông như ống).
- Mức object: E128 / E134 có train3 tăng đều (0,63 → 0,75) khi val dừng 0,66–0,68 và giảm từ epoch 6–7 (F198, F212).
- Augment / bỏ đặc trưng trang / loss cặp / cân bằng PDF không thu hẹp khoảng cách (E123–E126, E129, E132; F184–F211).
- Nhánh ảnh luôn là phần **cộng thêm** (zero-init) vào token vector; transformer vector ra quyết định. E131 thêm đọc ảnh theo tiếp tuyến / hai
  bên nét vào E114: best vẫn epoch 0 (F209) — E114 đã khớp train qua token nên phần ảnh không còn tín hiệu học.
- OpenOBB25: gán nét bằng luật hình học từ hộp ảnh 0,579 > đầu members học từ token 0,508 (F71–F73): phần ảnh tốt, phần học từ token kém.

## 2. Giả thuyết

**H153:** token vector mang "dấu vân tay" mã hoá của từng file CAD (giá trị màu / độ dày chính xác, cấu trúc item, thứ tự vẽ, thuộc tính payload).
Với 19 trang train, đường này khớp train gần hết nên đường ảnh không được học (học lối tắt, cạnh tranh giữa hai nguồn đầu vào). Ẩn các trường đó,
chỉ giữ hình học mà ảnh cũng thấy, thì quyết định phải đi qua ảnh: khoảng cách train / val thu hẹp và kết quả trên dự án lạ (Viewer8) giữ hoặc tăng.

## 3. Kiến trúc và luồng dữ liệu (bước 1)

Cùng mạng openstroke28 (E114: U-Net transformer trên nét + nhánh ảnh ResNet18 + FPN stride 4, không layer), khởi tạo mới (ResNet18 ImageNet).
Biến duy nhất: `feature_drop=image_first`.

```
ảnh vẽ lại từ nét (độ đậm theo màu, đúng độ dày)  → ResNet18 + FPN → đặc trưng lấy dọc từng nét ┐
16 điểm lấy mẫu dọc nét (hình dạng, PointNet)                                                   ├→ transformer theo vị trí → logit nét
10 đặc trưng hình học: độ dài, khung bao, độ dẹt, hướng, độ thẳng, đầu khép                     ┘
ẩn (37): màu RGB, độ dày, loại s/f/fs, có nét / có tô, opacity, even-odd, dashed, closePath, số đoạn, tỷ lệ l/c/re/qu, thứ tự vẽ,
         thống kê layer / kiểu nét của trang, mật độ lân cận
```

Màu và độ dày vẫn tới model **qua ảnh** (như mắt người thấy nét đậm / nhạt, dày / mảnh), không qua giá trị số chính xác của file.

**Bước 2 (chưa code, chỉ khi H153 đạt):** (a) nhìn thô: ảnh độ phân giải thấp cả cửa sổ → vùng giống ống; (b) zoom: vẽ lại vùng đó ở 3–4 px / pt;
(c) đánh nét: quyết định từng nét từ đặc trưng zoom dọc nét, gom object theo mảnh cắt tại đầu mút chạm (OpenStroke34) với cạnh cũng đọc từ ảnh zoom.

**Nguồn SOTA:** CADTransformer / VecFormer (lấy đặc trưng ảnh tại primitive; lấy: đọc ảnh dọc nét; bỏ: token style làm đặc trưng chính).
PointRend (Kirillov et al., CVPR 2020: chỉ tính lại ở độ phân giải cao tại điểm không chắc) và Recurrent Attention Model (Mnih et al., 2014: nhìn
lướt rồi zoom) cho bước 2. Học lối tắt (Geirhos et al., 2020) và cạnh tranh giữa nguồn đầu vào (Wang, Tran, Feiszli, CVPR 2020) giải thích vì
sao thêm nguồn vector làm kém tổng quát. Không chép code ngoài.

**Tận dụng OpenOBB cho phần nhìn ảnh (user 2026-11-23).** OpenOBB đã nhận diện tốt hình dạng ống gió và bắt đầu mút / khớp hộp. Bước 1 thêm nhánh
E136: thay ResNet18 ImageNet bằng **backbone HGNetv2-B5 của một detector OpenOBB HVAC**, đọc ảnh ở đúng tỉ lệ OpenOBB đã học (4 px / pt, giới hạn bởi
`img_max_px`), vẫn `image_first`. Đó là "con mắt đã biết ống gió" của OpenOBB, còn vector chỉ chỉ chỗ.

- Detector nguồn phải sạch: mọi detector HVAC cũ (E68a, E69, E106) train trên obb_hvac7 = 18 trang GT + **25 trang nhãn pipeline**, và E111 có Viewer
  chưa rà; AGENTS 0.2 cấm cả hai trong train. GT5 không nằm trong train của chúng (đã kiểm `pages.json`), nhưng nhãn giả thì có. Vì vậy **E136d**
  train lại detector OpenOBB chỉ ảnh (`openobb24x`, khởi tạo COCO công khai) trên **GT-only remote 18 / 6** (`openstroke.obb.gt_dataset`, val gồm
  GT5), 6 lớp. Nếu user cho phép dùng E106 làm nguồn (biết là có nhãn pipeline), chỉ cần đổi `img_weights`.
- Số phụ của E136d: F2 hộp rule_nl trên 6 trang val, so với E106 0,629 (có nhãn pipeline) để biết detector sạch còn tốt đến đâu.

## 4. Triển khai

- `openstroke/features.py`: `GEOMETRY`, `FEATURE_GROUPS["image_first"]` (37 trường ẩn, 10 giữ). Mặt nạ lưu trong checkpoint (`feat_keep`), infer
  ẩn y hệt.
- `openstroke/engine/trainer.py`: mọi run GT-only (không đầu vùng) in `Stroke F2 train3 x / val y` mỗi epoch, cùng ngưỡng.
- `openstroke/models/image_branch.py`: `obb_backbone_state` nạp `img_weights` từ cả checkpoint OpenOBB (`{"model": ...}`, lấy `backbone.*`) lẫn
  state dict backbone trần (E48). `openstroke/engine/trainer.py`: model khởi tạo mới nhận `img_backbone` / `img_weights` (trước chỉ khi đổi backbone
  trên checkpoint cũ). Trọng số backbone nằm trong checkpoint sau train; infer không cần file OpenOBB.
- `tests/test_os38.py`: nhóm `image_first` giữ đúng 10 trường hình học, ẩn mọi trường mã hoá; model nhận mặt nạ; backbone từ checkpoint OpenOBB
  vào nhánh ảnh không đổi giá trị.

## 5. Dữ liệu, split, metric

- e12, GT-only hvac10b: 19 trang AI_Takeoff GT train, val GT5 (doc180, 187, 189, 200, 201), seed 0, như E128–E134. Không dùng Viewer trong train.
- Chọn best: mean F2 nét theo PDF GT5 (như GT-only hiện có). Báo thêm: F2 train3 cùng ngưỡng (khoảng cách học thuộc), F2 object gom theo nét
  (`OBJECTS strokes`), và chấm cặp E135a / E135b trên GT5 + Viewer8 (Viewer8 = 8 trang dự án khác, nhãn một phần, ngưỡng chung 0,20, TTA ×1).

## 6. Baseline và ablation

| Run | Khởi tạo | Đặc trưng | Epoch |
|---|---|---|---|
| E135a (đối chứng) | mới, ResNet18 ImageNet | `layer_stats` ẩn (công thức E114) | 20 |
| **E135b** | mới, ResNet18 ImageNet | `image_first` | 20 |
| **E135c** | mới, ResNet18 ImageNet | `image_first` + 171 trang Viewer train của e12 (`viewer_share=0.65` như E114) | 20 |
| E136d (detector nguồn) | OpenOBB `openobb24x` COCO | chỉ ảnh, GT-only 18 / 6, 6 lớp | 40 |
| **E136** | mới, HGNetv2-B5 từ E136d, 4 px / pt | `image_first` | 20 |

E114 (train trên 190 trang có Viewer) chỉ là mốc lịch sử, khác dữ liệu, không dùng để chấm H153.

## 7. Kỳ vọng (ước lượng, chưa đo)

E135a: mean F2 PDF GT5 0,85–0,90 (mới, chỉ 19 trang). E135b: train3 thấp hơn E135a 0,02–0,04; GT5 trong ±0,01; Viewer8 F2 nét +0,01 đến +0,04.

E136: Viewer8 F2 nét +0,01 đến +0,03 so với E135b; GT5 ±0,01; khoảng cách train / val không lớn hơn E135b. E136d: rule_nl 0,58–0,63.

## 8. Tiêu chí (ghi trước GPU)

Khoảng cách = F2 train3 − mean F2 PDF GT5 ở best.pt (cùng ngưỡng, dòng `Stroke F2 train3` của epoch best).

- **Đạt:** khoảng cách E135b ≤ E135a − 0,02, **và** Viewer8 F2 nét E135b ≥ E135a, **và** GT5 E135b ≥ E135a − 0,01.
- **Bác bỏ:** khoảng cách E135b > E135a − 0,01, **hoặc** Viewer8 F2 nét E135b < E135a − 0,02.
- Ở giữa: vùng xám; xem từng trang Viewer và F2 object trước khi quyết bước 2.

**H155 (E135c so với E135b, biến: thêm 171 trang Viewer train, user 2026-11-23 nhắc dùng dữ liệu web gán nhãn):** nhãn Viewer Labels V1 chưa rà
(phủ thiếu, F186), chỉ vào train; chấm trên GT5 sạch và Viewer8 bằng cell chấm cặp, cùng giao thức với E135b.
- **Đạt:** mean F2 PDF GT5 ≥ E135b + 0,02 **hoặc** Viewer8 F2 nét ≥ E135b + 0,03.
- **Bác bỏ:** cả hai < E135b + 0,01.

**H154 (E136 so với E135b, biến: con mắt OpenOBB = backbone detector HVAC + tỉ lệ ảnh của nó):**
- **Đạt:** Viewer8 F2 nét E136 ≥ E135b + 0,02, **hoặc** mean F2 PDF GT5 ≥ E135b + 0,01; và khoảng cách train / val không lớn hơn E135b + 0,01.
- **Bác bỏ:** Viewer8 và GT5 đều < E135b + 0,005.
- H154 chấm độc lập với H153: nếu H153 bác bỏ mà H154 đạt, "ảnh quyết định" cần một con mắt mạnh hơn ResNet18 ImageNet.

## 9. Kiểm thử và chi phí

5 test (os38 + os33) qua trên CPU. Mỗi run ~2,4 phút / epoch như E114 → ~1 giờ cho 20 epoch; hai run chạy lần lượt hoặc hai runtime.

## 10. Lệnh Colab

Cell 2: cài lại code.

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

Cell 3: ZIP e12 đã giải nén ở `/content/os34/e12`. Cell 4 (đối chứng):

```python
!openstroke train \
    data=/content/os34/e12 \
    model=openstroke28 \
    epochs=20 \
    project=/content/drive/MyDrive/Train/OpenStroke38 \
    name=e135a_tokens \
    gt_only=true
```

Cell 5 (ảnh quyết định):

```python
!openstroke train \
    data=/content/os34/e12 \
    model=openstroke28 \
    epochs=20 \
    project=/content/drive/MyDrive/Train/OpenStroke38 \
    name=e135b_image_first \
    gt_only=true \
    feature_drop=image_first
```

Cell 6 (chấm cặp GT5 + Viewer8, sau khi cả hai xong; sửa tên thư mục nếu run tự đánh số):

```python
!python -m openstroke.compare_stroke_checkpoints /content/os34/e12 \
    /content/drive/MyDrive/Train/OpenStroke38/e135a_tokens/fold0/best.pt \
    /content/drive/MyDrive/Train/OpenStroke38/e135b_image_first/fold0/best.pt \
    --baseline-label E135a --candidate-label E135b \
    --out /content/drive/MyDrive/Train/OpenStroke38
```

Cell E135c (thêm trang Viewer; không có `gt_only`, nên train trên 190 trang của split hvac10b, val 13 trang):

```python
!openstroke train     data=/content/os34/e12     model=openstroke28     epochs=20     project=/content/drive/MyDrive/Train/OpenStroke38     name=e135c_image_first_viewer     feature_drop=image_first     viewer_share=0.65
```

Chấm cặp E135b / E135c trên GT5 + Viewer8:

```python
!python -m openstroke.compare_stroke_checkpoints /content/os34/e12     /content/drive/MyDrive/Train/OpenStroke38/e135b_image_first/fold0/best.pt     /content/drive/MyDrive/Train/OpenStroke38/e135c_image_first_viewer/fold0/best.pt     --baseline-label E135b --candidate-label E135c     --out /content/drive/MyDrive/Train/OpenStroke38
```

Cell 7 (E136d, detector OpenOBB chỉ ảnh, GT-only 6 lớp; dùng ZIP HVAC7 có sẵn trên Drive, nhãn pipeline bị loại khi dựng):

```python
!mkdir -p /content/hvac7 && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_hvac7_data.zip -d /content/hvac7
!python -m openstroke.obb.gt_dataset --source /content/hvac7/obb_hvac7 --out /content/openstroke15_gt
!python -c "from openstroke.obb.gt_dataset import ensure; print(ensure('/content/openstroke15_gt/data.yaml', 'hvac6'))"
!openstroke obb train     data=/content/openstroke15_gt_6/data.yaml     model=openobb24x     tile=True imgsz=1024 tile_scale=1.0     epochs=40     project=/content/drive/MyDrive/Train/OpenStroke38     name=e136d_obb_gt
```

(Nếu ZIP nằm ở `MyDrive/Dataset/OpenStroke9/`, sửa đường dẫn unzip. Dòng in của lệnh `ensure` cho biết thư mục 6 lớp; dùng đúng thư mục đó ở `data=`.)

Cell 8 (E136, sau khi E136d xong):

```python
!openstroke train     data=/content/os34/e12     model=openstroke28     epochs=20     project=/content/drive/MyDrive/Train/OpenStroke38     name=e136_obb_eye     gt_only=true     feature_drop=image_first     img_backbone=hgnetv2_b5     img_weights=/content/drive/MyDrive/Train/OpenStroke38/e136d_obb_gt/weights/best.pt     img_ppt=4
```

Cell 9 (chấm cặp E135b / E136 trên GT5 + Viewer8):

```python
!python -m openstroke.compare_stroke_checkpoints /content/os34/e12     /content/drive/MyDrive/Train/OpenStroke38/e135b_image_first/fold0/best.pt     /content/drive/MyDrive/Train/OpenStroke38/e136_obb_eye/fold0/best.pt     --baseline-label E135b --candidate-label E136     --out /content/drive/MyDrive/Train/OpenStroke38
```

Thứ tự: E135a, E135b, E136d chạy độc lập (song song nếu đủ GPU); E136 sau E136d; chấm cặp sau cùng. Gửi log từng cell là đủ.

## 11. Kết quả thật từ log

**E135b** (log user 2026-11-23, 0,63 h, khởi tạo mới, ResNet18 ImageNet, `image_first`, GT-only 19 / 5, 20 epoch). best.pt epoch 17.

| Epoch | train3 | val mean F2 PDF (TTA ×1) | Khoảng cách |
|---|---|---|---|
| 1 | 0,698 | 0,615 | 0,083 |
| 6 | 0,799 | 0,747 | 0,052 |
| 10 | 0,824 | 0,760 | 0,064 |
| 14 | 0,840 | 0,765 | 0,075 |
| **17 (best)** | 0,848 | 0,765 | **0,083** |
| 20 | 0,863 | 0,764 | 0,099 |

- best.pt + TTA ×8, ngưỡng 0,50: mean F2 PDF **0,785**, worst 0,690 (doc189); micro F2 0,811 (R 0,821 P 0,772). Từng trang: doc180 0,743
  (R 0,717), doc187 0,893, doc189 0,690 (P 0,411), doc200 0,780, doc201 0,819.
- Val bão hoà từ epoch 8 (~0,76) trong khi train3 tăng tới 0,863: khoảng cách nới từ 0,05 lên ~0,10, nhưng val không giảm (khác E128 / E134).
- Mốc lịch sử E114 (190 trang có Viewer, khác dữ liệu): mean F2 PDF GT5 0,906. H153 chờ E135a và chấm cặp Viewer8.

## 12. Điểm yếu và hướng sau

- Bước 1 chỉ kiểm nguồn học thuộc, chưa mô phỏng zoom. Nếu đạt: bước 2 (nhìn thô → zoom → đánh nét, PointRend-like) thành version riêng.
- Nếu bác bỏ: học thuộc không đến từ token mà từ chính dữ liệu hẹp; hướng chính là dữ liệu Viewer HVAC đã rà (`docs/VIEWER_HVAC_DATA.md`).
- Khởi tạo mới trên 19 trang có thể yếu hơn E114 ở số tuyệt đối; H153 chỉ so E135b với E135a.
