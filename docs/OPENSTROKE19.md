# OpenStroke19: phân lớp + gom từng nét (panoptic trên nét) chạy cạnh E69, bắt đầu đúng bằng E69

Giả thuyết: **H119**. Mốc: **E69 0,511** (members, native, 7 trang); E101 (v17) 0,503. Probe 0 GPU (GT + dự đoán đã lưu của E69, không suy luận):
`research/probes/e69_error_probe.py`, `research/probes/stroke_semantic_ceiling.py`, `research/probes/e69_regroup_probe.py`.
Trạng thái: **code + test xong (101 test pass), chưa train.** Code: `openstroke/obb/models/stroke_panoptic.py` (`StrokePanoptic`, `panoptic_loss`, `PanopticPage`, `panoptic_objects`), `openstroke/obb/models/openobb25.py` (`stroke_panoptic`), `openstroke/obb/predict.py` (gộp xác suất và tâm của một nét qua các ô theo id), `openstroke/obb/object_validation.py` (chấm `panoptic` là chính, `members` cạnh bên). Test: `tests/obb/test_v15.py::test_v19_*`, `tests/obb/test_object_selection.py[v19]`.

## 1. Bằng chứng: điểm yếu thật của E69

**(a) Object HVAC là tập rất ít nét.** 2.084 object GT trên 7 trang:

| Số nét / object | Số object | E69 khớp | E69 bỏ sót | E69 gộp |
|---|---|---|---|---|
| 1 nét | 803 (39 %) | 63 % | 17 % | 13 % |
| 2 nét | 382 (18 %) | 40 % | 29 % | 19 % |
| 3–4 nét | 671 (32 %) | 50 % | 21 % | 12 % |
| 5–10 nét | 220 (11 %) | 29 % | 27 % | 11 % |
| > 10 nét | 8 | 25 % | 50 % | 12 % |

89 % object có ≤ 4 nét; 39 % object **chính là một nét** (reducer, chân rẽ, đoạn ống vẽ bằng một shape). Object nhỏ (< 200 px) chiếm 79 % và
khớp 44 %; 77 % object nhỏ chạm object khác trong 2 pt (F100).

**(b) Với bài toán này, nhãn từng nét là gần đủ.** Nếu mỗi nét có đúng lớp của object chứa nó, gom nét cùng lớp nằm sát nhau (thành phần
liên thông, không học) đã cho:

| Gom nét cùng lớp trong bán kính | F2 | R | P |
|---|---|---|---|
| 1 px | 0,646 | 0,786 | 0,378 |
| 4 px | 0,796 | 0,864 | 0,605 |
| **8 px** | **0,848** | 0,874 | 0,759 |

Với gom học được (mỗi nét dự đoán tâm object của nó, kiểu PointGroup), trần là 1,000.

**(c) Lớp nét của E69 là chỗ yếu, không phải cách gom.** Lấy chính lớp nét mà E69 suy ra từ hộp của nó, gom lại theo khoảng cách thay vì
theo hộp: 0,449–0,470, thấp hơn gom theo hộp 0,506. Tức là E69 không biết đúng nét nào thuộc lớp nào; đổi cách gom không cứu được.

**(d) Vì sao E69 học lớp nét kém:** đầu members của E69 chỉ học trên nét nằm trong cổng của hộp đã khớp GT (dạy theo hộp đúng), không bao giờ
học "nét này có thuộc object nào không, lớp gì" trên toàn ô. 18 trang train có ~2,7 nghìn object nhưng **~3,9 triệu nét trong ô**: giám sát
từng nét nhiều hơn giám sát từng object khoảng ba bậc. Với ít dữ liệu, đây là đòn bẩy lớn nhất.

**(e) Lần thử lớp nét trước (v10–v12) không phải phép thử công bằng:** đầu đồ thị khi đó có lớp nét đúng 0,12–0,18 (F95), nhưng train từ E85
(detector đã tụt), lr 1e-3 ở v12, loss chia đều dương / âm, và lớp object lấy theo lớp nét đa số trong thành phần liên thông của cạnh k-NN
(precision cạnh 0,03–0,07). v19 không có đồ thị cạnh: gom bằng tâm dự đoán.

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v19 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Object = 1–4 nét; nhãn từng nét gần đủ (trần 0,848) | **Đầu lớp nét dày đặc**: mỗi nét (token vector đã trộn ảnh) → 9 lớp HVAC + nền, học trên **mọi nét của ô** (CE, nét "không biết" bỏ qua) | lớp nét tốt hơn lớp nét suy từ hộp của E69 | F2 panoptic ≤ 0,53 |
| Gom bằng khoảng cách trần 0,848; gom theo tâm trần 1,000 | **Đầu tâm object**: mỗi nét của object dự đoán độ lệch tới tâm object (L1); giải mã = nét cùng lớp có tâm dự đoán gần nhau (≤ bán kính chọn trên val) | tách được hai object cùng lớp nằm sát | — |
| Phải xuất phát từ bản tốt nhất và bắt đầu đúng bằng nó | Thân, detector, members của E69 giữ nguyên; hai đầu mới là nhánh riêng, ra object nguồn `panoptic`; đường `members` của E69 vẫn chấm song song | members ở epoch 1 ≈ E69 | members epoch 1 < 0,45 |
| Đầu mới học từ 0 trên mô hình đã train | LR ×10 cho đầu mới (`new_lr_mult`) | — | — |

Nguồn (AGENTS mục 10): SymPoint (Liu và cộng sự, ICLR 2024) và CADSpotting (2024) coi mỗi primitive CAD là một điểm có lớp ngữ nghĩa và
thuộc một instance (panoptic symbol spotting); PointGroup (Jiang và cộng sự, CVPR 2020) gom điểm bằng độ lệch tới tâm + thành phần liên
thông theo lớp. Lấy: giám sát ngữ nghĩa trên từng primitive và gom theo tâm dự đoán. Bỏ: bộ giải mã truy vấn kiểu Mask2Former (v15 / v16 cho
thấy học từ đầu trên 18 trang không đủ), dữ liệu FloorPlanCAD, số PQ của các bài đó (khác bài toán HVAC, không dùng làm kỳ vọng). Thay đổi
cho HVAC không layer: token nét không có cột layer; đặc trưng ảnh lấy tại nét từ E69; lớp object HVAC 9 lớp; giải mã trên cả trang sau khi
gộp xác suất của một nét qua các ô.

## 3. Kỳ vọng (ghi TRƯỚC khi train), con số tuyệt đối

Run E103: khởi tạo E69, chỉ GT 18 / 8 trang, 20 epoch, lr0 1e-4 (đầu mới ×10), các mặc định khác như E101. Chấm native, 7 trang.

| Chỉ số | E69 | E101 (v17) | **Đạt nếu (E103)** | **Bác bỏ nếu** |
|---|---|---|---|---|
| F2 object, đường `panoptic` | — | — | **≥ 0,60** | **≤ 0,53** |
| Recall, `panoptic` | — | — | ≥ 0,62 | — |
| F2 object, đường `members` (đối chứng trong cùng run) | 0,511 | 0,503 | 0,48–0,53 (không tụt) | < 0,45 |
| F2 epoch 5, `panoptic` | — | — | ≥ 0,50 | — |

Mục tiêu sau khi đạt: tăng độ chính xác lớp nét (đo F2 chiều dài nét theo lớp) tới mức cho F2 object 0,75 (trần gom đơn giản 0,848).

## 4. Điều chứng minh sai

- `panoptic` ≤ 0,53: giám sát dày đặc không làm lớp nét tốt hơn mức E69 tự có; khi đó nút thắt nằm ở đặc trưng (thân), không ở đầu ra.
- `panoptic` ≥ 0,60 nhưng `members` tụt < 0,45: hai đầu tranh nhau thân; tách LR / tách loss.

## 5. Lệnh (Colab)

Cần checkpoint E69 trên Drive (`MyDrive/Train/OpenStroke9/e69/best.pt`, upload từ `F:\Source_code\OpenStroke\runs_colab\e69\best.pt`).
Trên VM mới: mount Drive, `!pip install -q "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"`, giải nén ZIP như các run trước, rồi:

```python
!openstroke obb train \
  data=/content/openstroke9_e83/obb_hvac7/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke9/e69/best.pt \
  stroke_panoptic=True \
  epochs=20 \
  project=/content/drive/MyDrive/Train/OpenStroke19 \
  name=e103_panoptic
```

Mỗi epoch log in dòng `object F2` của đường `panoptic` (chính) và dòng thụt lề có cả `panoptic` và `members`. Cột loss có `panoptic_cls`,
`panoptic_off`. Giải mã: nét có xác suất lớp tốt nhất ≥ 0,3, gom theo tâm dự đoán trong 24 px trang; ngưỡng điểm object chọn trên val như
mọi đường.

## 6. Kết quả thật

(trống)
