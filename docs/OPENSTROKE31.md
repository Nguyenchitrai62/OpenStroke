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

**E122** (log user 2026-11-18). Lệch thẻ: chạy **15 epoch** thay vì 10 như E121, nên lịch LR cosine khác đối chứng. 4:15 / epoch, 82,4 GB
(bằng E120), 1,17 h.

- F2 nét val mỗi epoch (chưa TTA): 0,830 / 0,833 / 0,839 / 0,841 / 0,846 / 0,847 / **0,847** / 0,847 / 0,846 / 0,847 / 0,847 / 0,846 / 0,844 /
  0,844 / 0,842. Đỉnh của E120 là 0,853 (epoch 6), E121 bám E120 trong ±0,005: E122 thấp hơn khoảng 0,006 ở mức từng epoch.
- Từ epoch 11 model đổi R lấy P: R 0,90 → 0,88, P 0,69 → 0,725, ngưỡng tốt nhất lên 0,75–0,80.
- `best.pt` = epoch 7, F2 nét + TTA ×8 **0,850** (R 0,891, P 0,717, ngưỡng 0,75); E121 0,853 (R 0,911, P 0,679), E120 0,857 (R 0,907, P 0,703).
  Qua sàn F2 nét ≥ 0,843 của §8, nhưng R thấp hơn E121 0,02.
- Theo trang: GT 0,895–0,957 (R 0,949–0,977); Viewer 0,627–0,863. Viewer có hai kiểu lỗi: R thấp ở `vw_579f`, `vw_e241` ×2, `vw_2f3d`
  (0,65–0,71), và P thấp ở `vw_62b1` p2 / p3 (0,37 / 0,26), như E120 / E121 (F171).
- `edge_loss` (train) 1,11 → 0,48, chưa phẳng tới epoch 15; ở epoch 7 (`best.pt`) là 0,57.

**H136, chấm bằng `edge_group_probe.py`** (sổ cái F177; `runs_colab/e122/edge_group_{best,best_hi,last}.txt`, E121 chấm lại cùng lưới
`runs_colab/e121/edge_group_probe_{g7,hi}.txt`). Lớp của object lấy từ GT; thước đo IoU chiều dài nét ≥ 0,5, không phải rule_nl. Mỗi ô lấy ngưỡng
cạnh tốt nhất trong {0,05; 0,1; 0,2; 0,3; 0,5; 0,7; 0,8; 0,9; 0,95} (chọn trên chính val, cả hai run cùng cách). Nét model chọn ở ngưỡng checkpoint
của từng run (E121 0,60, E122 0,75). `best.pt` TTA ×8; `last.pt` TTA ×1, chỉ 5 trang GT.

| Nét / cách gom (GT5 / Viewer 8) | E121 | E122 `best.pt` | E122 `last.pt` (GT5) |
|---|---|---|---|
| Nét GT, lớp đúng + cạnh | 0,912 / 0,810 | **0,962 / 0,925** | 0,958 |
| Nét GT, chỉ cạnh (không dùng lớp) | 0,798 / 0,751 | **0,899 / 0,889** | 0,904 |
| Nét model, lớp đúng + cạnh | 0,863 / 0,733 | **0,881 / 0,744** | 0,871 |
| Nét model, chỉ cạnh | 0,718 / 0,669 | **0,820 / 0,712** | — |
| Nét model, lớp đúng + gom nét chạm (không dùng cạnh) | 0,833 / 0,714 | 0,834 / 0,704 | 0,826 |
| Nét GT, cạnh oracle (trần) | 0,984 / 0,985 | 0,984 / 0,985 | 0,984 |

Đầu cạnh trên cặp nét đều thuộc object GT, ngưỡng 0,3: R 0,90–0,98 / P 0,68–0,91 trên trang GT (E121 ở 0,1: R 0,30–0,86 / P 0,33–0,65). Trang yếu
nhất `vw_e3fe` (R 0,63, P 0,73).

- **H136 đạt theo tiêu chí ghi trước:** nét GT + lớp 0,962 ≥ 0,94; nét model + lớp 0,881 ≥ 0,88; Viewer không tụt (+0,115 / +0,011); F2 nét 0,850 ≥ 0,843.
- Giới hạn của kết luận: dòng nét model + lớp chỉ vượt mốc 0,001 (trong nhiễu của 5 trang); ngưỡng cạnh chọn trên val; E122 train 15 epoch so với 10
  của E121, nên phần tăng chưa tách được giữa loss mới và train lâu hơn.
- Đầu cạnh giờ tự tách được object khi không có lớp: nét GT 0,798 → 0,899, nét model 0,718 → 0,820 (GT5). Đây là dòng gần pipeline thật nhất, vì
  pipeline không có lớp oracle.
- `last.pt` không gom tốt hơn `best.pt` (0,958 so với 0,962, khác TTA): chọn checkpoint theo F2 nét không làm hại đầu cạnh ở run này.
- Trên nét GT, gom theo cạnh đã gần trần (0,962 so với 0,984). Trên nét model, khoảng cách tới nét GT là 0,08 (GT5) và 0,18 (Viewer): **nút thắt
  chuyển sang chọn nét** (nét thừa trên Viewer `vw_62b1`, R thấp ở 4 trang Viewer).

## 12. Kết luận và version tiếp theo

H136 đạt (§11). Đầu "cùng object" của E122 là tín hiệu gom object tốt nhất hiện có; hậu kỳ hiện tại của app chưa dùng nó.

1. **Hậu kỳ (0 GPU, việc kế tiếp):** dựng object = thành phần liên thông của nét chọn theo cạnh E122, đặt tên lớp bằng bộ phân loại hình dạng
   của khối (`hvac_faces`) hoặc hình học tập nét; chọn ngưỡng cạnh trên trang GT train (`openstroke scores` E122 trên 24 trang GT), rồi chấm F2
   object rule_nl trên val so với E120 0,827 / 0,544 (F172). Đây là số so được với mục tiêu 0,80.
2. **Model:** nút thắt chuyển sang chọn nét (nét model + lớp + cạnh 0,881 so với nét GT 0,962 trên GT5; 0,744 so với 0,925 trên Viewer).
   R của E122 thấp hơn E121 0,02; version sau phải giữ R.
3. Nếu cần tách phần do loss và phần do train 15 epoch: chạy E121 15 epoch làm đối chứng.
