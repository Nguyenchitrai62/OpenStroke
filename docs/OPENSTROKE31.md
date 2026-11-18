# OpenStroke31 — gom object bằng đầu "cùng object": loss cân bằng trên cặp nét đúng

Ngày: 2026-11-18. Trạng thái: đã code và test CPU, chưa train. Sổ cái F175, giả thuyết H136.

## 1. Mục tiêu và phạm vi

Mục tiêu chung không đổi: F2 object ≥ 0,80 trên GT giữ ngoài train, rồi tăng mạnh hơn mức đó. Model nét vẫn là một mắt xích của pipeline:
model chọn nét, hậu kỳ dựng object. Version này nhắm **bước gom nét thành object**, không nhắm việc chọn nét.

Input không layer như openstroke28 / 30. Không đổi mạng, dữ liệu, split hay cách chấm. Chỉ đổi loss của đầu "cùng object" có sẵn từ openstroke30.

## 2. Bằng chứng và điểm yếu version trước

**Model nét đã bão hoà.** F2 nét val qua 5 run: E114 0,839 (10 và 20 epoch như nhau), E116 0,840, E117 0,842, E120 0,857, E121 0,853.

**Trần của cách dựng object hiện tại thấp hơn nhiều so với trần của gom theo cạnh.** Hậu kỳ "khối khép kín + logic" với nét hoàn hảo cho GT5
0,854 / Viewer 0,640 (F165); với lớp từng nét đúng 0,853–0,868 (F160, F167). E120 hiện đạt 0,827 / 0,544 (rule_nl, F172).

**Probe F175** (`research/probes/edge_group_probe.py`, 13 trang val hvac10b, 0 GPU). Object = thành phần liên thông trên đồ thị cặp nét ứng viên
của model (`eij`, cạnh nối đầu mút / khe nét đứt / chữ T). Lớp object lấy từ GT, nên chỉ phần gom được đo. Chấm bằng IoU chiều dài nét ≥ 0,5,
cùng lớp (thước đo F139, không phải rule_nl). Đối chứng là E121 (`runs_colab/e121/edge_group_probe.txt`), mỗi ô lấy ngưỡng cạnh tốt nhất
trong lưới {0,05; 0,1; 0,2; 0,3; 0,5}:

| Nét | Cách gom | GT5 | Viewer 8 |
|---|---|---|---|
| Nét GT | cạnh oracle (giữ cạnh khi hai đầu cùng object GT) | **0,984** | **0,985** |
| Nét GT | lớp từng nét đúng + gom nét chạm nhau | 0,909 | 0,870 |
| Nét GT | chỉ cạnh E121 | 0,798 | 0,751 |
| Nét GT | lớp đúng + cạnh E121 | 0,912 | 0,810 |
| Nét model (p ≥ 0,60) | lớp đúng + gom nét chạm nhau | 0,833 | 0,714 |
| Nét model (p ≥ 0,60) | lớp đúng + cạnh E121 | 0,863 | 0,733 |

- Đồ thị ứng viên đủ để dựng gần đúng mọi object (0,98). Gom theo cạnh có trần cao hơn hẳn mọi cách gom đang dùng.
- Đầu cạnh của openstroke30 gần như không thêm gì so với gom nét chạm nhau: 0,912 so với 0,909 trên nét GT.
- Trên cặp nét cùng thuộc object GT, đầu cạnh ở ngưỡng 0,5 chỉ bắt 22–44 % cặp cùng object (GT5); phải hạ ngưỡng xuống 0,1 mới bắt 30–86 %,
  khi đó precision còn 0,33–0,65. Đầu cạnh phân tách kém.
- Trên GT5, phần mất do chọn nét (0,909 → 0,833) và do gom (0,984 → 0,909) lớn ngang nhau, khoảng 0,075 mỗi phần.

**Vì sao đầu cạnh yếu.** `edge_loss` của openstroke30 là trung bình BCE trên mọi cạnh có ít nhất một nét đúng, trọng số 0,5. Phần lớn các cạnh đó
là nét đúng – nét nền, dễ phân biệt khi đầu nét đã biết nét nào là ống. Loại cạnh quyết định việc gom là **hai nét đúng chạm nhau nhưng thuộc hai
object khác nhau** (hai đoạn Duct nối tiếp, Duct chạm Elbow). Loại này chỉ chiếm phần nhỏ, nên gần như không ảnh hưởng tới loss: `edge_loss` của
E120 xuống 0,047 → 0,014 trong khi đầu cạnh vẫn không tách được object.

## 3. Giả thuyết đăng ký trước

**H136 (E122, loss):** khi loss của đầu "cùng object" cân bằng giữa cặp nét đúng cùng object và cặp nét đúng khác object, đầu cạnh học được
ranh giới giữa hai object chạm nhau, và gom theo cạnh trên nét GT vượt rõ gom nét chạm nhau. Biến duy nhất so với E121: `edge_pairs=true`.

## 4. Kiến trúc và luồng dữ liệu

Mạng giữ nguyên openstroke30, không thêm tham số:

```
nét (lệnh vẽ, màu, độ dày, điểm mẫu) + ảnh vẽ lại (ResNet18 + FPN)
  -> U-Net transformer cửa sổ (OpenStroke2)
  -> GraphAttention × 2 trên đồ thị nối
  -> đầu nhị phân                                -> logit nét
  -> EdgeHead(h_i + h_j, |h_i − h_j|, e_ij)      -> logit "cùng object" mỗi cạnh nối
```

Loss mới của đầu cạnh khi `edge_pairs=true` (`openstroke2.edge_loss(pairs=True)`). Đích t = 1 khi hai nét cùng đúng và cùng `oinst`:

```
L_edge = mean_BCE(cạnh nét đúng–nét đúng, cùng object)
       + mean_BCE(cạnh nét đúng–nét đúng, khác object)
       + 0,25 × mean_BCE(cạnh nét đúng–nét nền)
       (cạnh nét nền–nét nền: không có loss)
```

- Mỗi nhóm lấy trung bình riêng, nên nhóm "khác object" nặng bằng nhóm "cùng object" bất kể số cạnh. Nhóm vắng trong batch thì bỏ qua.
- Cạnh nét đúng–nét nền giữ trọng số nhỏ (0,25, hằng `EDGE_TB_WEIGHT`), để nét nhiễu model lỡ chọn không bị nối vào object thật.
- Cạnh nét nền–nét nền không mang loss: cả hai nét bị bỏ trước bước gom.
- Trọng số tổng của loss cạnh vẫn là `edge_loss=0.5` như openstroke30. Mức loss tăng tự nhiên vì cặp khó không còn bị cặp dễ pha loãng.
- Suy luận không đổi: `Predictor`, `openstroke scores` và `val_scores.npz` đã lưu logit cạnh (`<stem>__eij`, `__elog`) từ openstroke30.

**Nguồn SOTA (bài gốc):**
- GAT-CADNet (Zheng et al., CVPR 2022): dựng instance trong bản vẽ CAD bằng cách dự đoán cạnh nối hai primitive cùng instance trên đồ thị primitive.
  Lấy: dùng đầu cạnh làm tín hiệu instance chính, đã có từ openstroke30. Bỏ: đồ thị theo khoảng cách, cascade tinh chỉnh cạnh (đồ thị nối đầu
  mút của OpenStroke đã đủ: trần 0,98 ở F175).
- BCE cân bằng theo lớp là kỹ thuật chuẩn khi lớp hiếm bị lớp dễ lấn át. Ở đây cân bằng theo **loại cặp**, vì loại cặp quyết định việc gom.
  Không lấy PQ FloorPlanCAD của bài nào làm kỳ vọng cho F2 HVAC.
- Không chép code hay weights ngoài.

## 5. Cách triển khai

- `openstroke/models/openstroke2.py`: tham số `edge_pairs` (lưu trong config / checkpoint, không có trọng số), `edge_loss(..., pairs=)`,
  hằng `EDGE_TB_WEIGHT`.
- `openstroke/models/__init__.py`: `ARCHS["openstroke31"]` = openstroke30 + `edge_pairs`.
- `openstroke/engine/trainer.py`: truyền `pairs=model.edge_pairs` cho loss; bật `edge_pairs` trên checkpoint cũ không cần khởi tạo gì thêm.
- `openstroke/train.py`: cờ `edge_pairs=`, preset `model=openstroke31`. Một version nghiên cứu được fine-tune với công tắc của version mới hơn
  thì checkpoint lưu tên version mới (`describe_arch`, `arch_name`): E114 + `rel_attn=2 edge_head=true edge_pairs=true` lưu là `openstroke31`.

## 6. Dữ liệu, split, augmentation và metric

- Dữ liệu e11 như E121 (e10 + item vách + đường chéo của Elbow vuông), split hvac10b: train 19 trang GT + 171 trang Viewer, val 5 GT + 8 Viewer.
  Không augment thêm. Không có test riêng.
- **Metric chính của H136:** `edge_group_probe.py` trên `val_scores.npz` của run, cùng lưới ngưỡng cạnh với đối chứng E121:
  - `gt|edge+c`: nét GT, lớp đúng, gom theo cạnh. Đo riêng chất lượng đầu cạnh.
  - `pm|edge+c`: nét model ở ngưỡng checkpoint, lớp đúng, gom theo cạnh. Gom thực tế.
- Phụ: F2 nét val; R / P của cạnh cùng object trên cặp nét đều thuộc object GT; F2 object rule_nl với hậu kỳ mặc định của app (so được với
  E120 / E121, nhưng hậu kỳ mặc định chưa dùng cạnh nên không kỳ vọng đổi nhiều).

## 7. Baseline, ablation và recipe

| Run | Khởi tạo | Data | Cờ | So với |
|---|---|---|---|---|
| E121 (đối chứng, đã có) | E114 | e11 | `closure=0.5 rel_attn=2 edge_head=true` | — |
| **E122** | E114 | e11 | E121 + `edge_pairs=true` | E121 |

Cùng batch 8, lr0 0,001, warmup 1, 10 epoch như E121. Sàng lọc 10 epoch trước; chỉ chạy dài hơn nếu E122 qua tiêu chí.

## 8. Kỳ vọng và tiêu chí đạt / bác bỏ

Kỳ vọng (ước lượng của agent, chưa đo):
- `gt|edge+c` GT5 0,912 → 0,93–0,95; Viewer 0,810 → 0,83–0,86.
- `gt|edge` (không có lớp) GT5 0,798 → ≥ 0,85: đầu cạnh tự tách được object khác lớp.
- `pm|edge+c` GT5 0,863 → 0,87–0,89.
- F2 nét trong khoảng ±0,01 của E121 (0,853).

Tiêu chí (ghi trước khi train):
- **Đạt:** `gt|edge+c` GT5 ≥ 0,94 **và** `pm|edge+c` GT5 ≥ 0,88, Viewer không tụt quá 0,01 ở cả hai dòng, F2 nét ≥ 0,843.
  Khi đó version kế tiếp thay bước dựng object của hậu kỳ bằng gom theo cạnh + bỏ phiếu lớp, chấm F2 object rule_nl so với 0,827.
- **Bác bỏ:** `gt|edge+c` GT5 < 0,92, tức đầu cạnh vẫn không hơn gom nét chạm nhau (0,909) quá nhiễu; hoặc F2 nét tụt quá 0,01.
- Ở giữa: vùng xám, ghi số, không mở version dựa trên E122.

## 9. Kiểm thử và chi phí

`tests/test_os31.py` (4 test, CPU):
- loss cân bằng: bốn cạnh cùng object cộng lại nặng bằng một cạnh ranh giới; cạnh nét đúng–nét nền nặng 0,25; cạnh nét nền–nét nền không có loss;
  phần gradient của cạnh ranh giới lớn hơn ít nhất 2 lần so với loss openstroke30;
- đích đúng, nhóm vắng được bỏ qua;
- `openstroke31` = openstroke30 + `edge_pairs`, cùng tập trọng số; tên version khi fine-tune từ checkpoint đặt tên `openstroke28`;
- train tí hon qua trainer thật: loss được gọi với `pairs=True`, checkpoint giữ cờ, run khởi động lại từ checkpoint đó vẫn dùng loss mới.

Chi phí: không thêm tham số, không thêm lượt tính; dự kiến cùng thời gian và VRAM với E121.

## 10. Lệnh train Colab

Dữ liệu như E121: `MyDrive/Dataset/openstroke_e11_data.zip`.

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/os30 && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_e11_data.zip -d /content/os30
```

```python
!openstroke train \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    data=/content/os30/e11 \
    closure=0.5 \
    rel_attn=2 \
    edge_head=true \
    edge_pairs=true \
    batch=8 \
    lr0=0.001 \
    warmup_epochs=1 \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke31 \
    name=e122_edge_pairs
```

Gửi lại: log và `fold0/val_scores.npz` (có sẵn logit cạnh).

## 11. Kết quả thật từ log

Chưa train.

## 12. Kết luận và version tiếp theo

Chưa có kết quả. Nếu H136 đạt: thay bước dựng object bằng gom theo cạnh + bỏ phiếu lớp (bộ phân loại hình dạng hoặc đầu lớp), chấm F2 object
rule_nl. Nếu bác bỏ: đầu cạnh thiếu thông tin hình học ở chỗ nối (đầu mút gặp đầu mút thẳng hàng hay chữ T, có nét ngang cắt qua hay không),
thêm đặc trưng chỗ nối cho EdgeHead.
