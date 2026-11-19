# OpenStroke34 — đầu object trên mảnh nét, lan độ tin cậy qua các mảnh chạm nhau

Ngày: 2026-11-19. Trạng thái: đã code và test CPU, chưa train. Sổ cái F186–F187, giả thuyết H141.

## 1. Mục tiêu và phạm vi

Mục tiêu chung: F2 object ≥ 0,80 trên GT giữ ngoài train, model ra object kèm nét của từng object, hậu kỳ nhẹ. Version này sửa lỗi gốc của
kiến trúc nét (F186): **đơn vị token là cả item PDF**, trong khi một item thường vẽ nhiều object. Thêm một tầng thứ hai (đầu mảnh) đọc các mảnh
nét cắt tại góc và giao điểm, dự đoán mảnh nào là ống, lớp của mảnh và cạnh "cùng object" giữa các mảnh. Tầng 1 (model nét openstroke31) giữ
nguyên. Áp dụng chung cho dạng khối (ống gió HVAC) và nét mảnh (ống FIRE), vì đồ thị mảnh không phụ thuộc hệ.

## 2. Bằng chứng và điểm yếu version trước

- **Một item vẽ nhiều object (F186):** trên 24 trang GT, 19 % nét true (item) và **64 % chiều dài true** vẽ ≥ 2 object GT (polyline CAD chạy liền
  vách Duct → cung Elbow → Duct kế tiếp). Lớp và chỉ số object theo item (`ocls` / `oinst`) sai trên phần còn lại; đầu lớp E116 / E117 kém ở phụ
  kiện (Tee 0,34–0,40, Reducer 0,1–0,48, Chân rẽ 0,62–0,68, Duct 0,89); hậu kỳ phải đoán tên khối bằng 16 đặc trưng tay và bỏ khối nối (F179).
- **Mảnh sạch (F187):** cắt mỗi nét true tại chỗ đứt, góc ≥ 30°, chỗ chuyển thẳng ↔ cung, và chỗ nét true khác chạm vào (đỉnh hoặc thân đoạn):
  chiều dài vẽ ≥ 2 object còn **0,1 %** (item 61,9 %), 2,8 mảnh mỗi item; đoạn dài > 20 pt chỉ 1,5 % chiều dài nằm trên object thứ hai.
- **Trần trong giao thức mảnh** (val e12, nhãn, không model; `research/probes/piece_object_probe.py --oracle`, `scratchpad face_oracle.py`):

| Gom mảnh (mảnh ống và lớp đúng) | GT5 | Viewer 8 |
|---|---|---|
| Mọi cạnh chạm giữa mảnh cùng lớp | 0,875 | 0,782 |
| Chạm + đối diện, cùng lớp | 0,844 | 0,750 |
| Cạnh đúng (cùng object GT), chỉ cạnh chạm | 0,917 | 0,891 |
| **Cạnh đúng, cạnh chạm + đối diện** | **0,979** | **0,950** |

  Gom theo lớp không đủ (gộp object cùng lớp nằm sát nhau); cần cạnh "cùng object" học được, và cần cạnh **đối diện** giữa hai vách song song
  của ống (không chạm nhau): thiếu nó, ống có đầu thuộc phụ kiện bên cạnh bị tách đôi (trần 0,917).
- **Lỗi tự tin (F186):** 29–81 % chiều dài bắt thừa ở p ≥ 0,95; FN trang Viewer nằm cả cụm. User 2026-11-19: khi một số nét đã chắc là một
  object, các nét chạm liền kề nhiều khả năng cũng thuộc object đó, cho cả dạng khối và dạng nét mảnh.

## 3. Giả thuyết đăng ký trước

**H141 (E127):** đầu mảnh với cạnh chạm + đối diện và lan độ tin cậy dựng object tốt hơn gom theo lớp, tiến gần trần của cạnh đúng. Biến so với
E122: thêm `piece_head=true` (dataset e12 = e11 + mảnh), fine-tune từ E122.

## 4. Kiến trúc và luồng dữ liệu

```
tầng 1 (openstroke31, không đổi): nét → U-Net transformer + nhánh ảnh + attention đồ thị → logit nét h_i, embedding h
tầng 2 (mới, PieceHead):
  mảnh p của nét ứng viên (pieces.shape_pieces)            đặc trưng mảnh (22): 8 điểm quanh tâm / độ dài, log độ dài, cong,
  z_p = MLP(đặc trưng mảnh) + W h_{nét cha} + W' σ(logit nét cha)   hướng (góc kép), lệch tâm so với nét cha
  đồ thị mảnh: cạnh chạm (đầu mút ≤ 0,5 pt, kể cả chữ T; 4 gần nhất mỗi đầu) + cạnh đối diện (song song ≤ 10°, cách 1–100 pt,
               chồng hình chiếu ≥ 30 %, 2 gần nhất)
  3 vòng: m_{j→i} = σ(gate([z_i, z_j, geo_ij, c_j])) · MLP([z_i, z_j, geo_ij, c_j]); z_i ← LN(z_i + mean_j m_{j→i}); c = σ(fg(z))
          (độ tin cậy c của mảnh gửi làm cổng; mỗi vòng đọc c vừa cập nhật: mảnh chắc chắn kéo mảnh chạm / đối diện)
  ra: logit ống mỗi mảnh, logit lớp (10 lớp HVAC), logit "cùng object" mỗi cạnh
object = mảnh ống nối theo cạnh cùng object, cùng lớp dự đoán
```

- Mảnh ứng viên (khi train và chấm val): nét true + nét chạm nét true + mẫu nét cùng kiểu (âm khó). Mảnh chỉ bị cắt bởi đầu mút của nét true
  (khi chạy thật: nét tầng 1 chọn), không bởi gạch cách nhiệt chạm vách.
- Nhãn mảnh: object GT có cạnh gần nhất dọc mảnh (F180); `pc_fg` = nét true và ≥ 1/2 mảnh nằm trên object đó; lớp chung chỉ trên trang GT (trang
  Viewer lớp = −1, vì quy ước lớp khác, F162 / F167); chỉ số object.
- Loss: BCE ống (√(độ dài + 1), pos_weight 2) + 0,5 × CE lớp + 0,5 × loss cạnh cân bằng (cùng / khác object giữa mảnh ống, 0,25 cho mảnh ống–nền),
  trọng số chung `piece_loss=1`.

**Nguồn SOTA (bài gốc):**
- GAT-CADNet (Zheng et al., CVPR 2022): đồ thị primitive, attention có quan hệ cạnh, dự đoán cạnh cùng instance. Lấy: dự đoán cạnh cùng instance
  trên đồ thị. Thay: primitive là mảnh cắt tại góc / chỗ chạm (không phải cả item, F186); cạnh theo chạm và đối diện (không theo khoảng cách thuần).
- CADTransformer / SymPoint-V2 (Fan et al. CVPR 2022; Liu et al. 2024): primitive là đoạn / cung lấy từ lệnh vẽ, không gom cả path. Lấy: đơn vị cấp
  lệnh vẽ. Giữ: tầng 1 theo item cho bước chọn nét (đã tốt, F2 nét 0,85), tầng 2 chỉ trên nét ứng viên để chi phí nhỏ.
- Lan nhãn qua đồ thị có cổng theo độ tin cậy (họ label propagation / iterative refinement): mỗi vòng đọc lại dự đoán của vòng trước. Theo đề
  xuất của user 2026-11-19.
- Không chép code ngoài.

## 5. Cách triển khai

- `openstroke/pieces.py`: `shape_pieces`, `piece_objects`, `facing_edges`.
- `tools/make_e12.py`: dataset e12 (e11 + `pc_shape`, `pc_pts`, `pc_len`, `pc_cur`, `pc_edges`, `pc_ekind`, `pc_fg`, `pc_cls`, `pc_inst`).
- `openstroke/data.py`: `attach_pieces`, `window_pieces`, `piece_features`, `piece_batch`; augment / TTA biến đổi điểm mảnh cùng nét.
- `openstroke/gpu_batch.py`: mảnh trong batch dựng trên GPU (chấm val); train có đầu mảnh dùng batch CPU (augment numpy).
- `openstroke/models/openstroke2.py`: `PieceHead`, `piece_loss`, công tắc `piece_head`; `models.ARCHS["openstroke34"]`.
- `openstroke/engine/trainer.py`: loss, khởi tạo đầu mảnh mới trên checkpoint cũ, đầu ra mảnh vào `val_scores.npz` (`<stem>__pid / __pfg /
  __pcls / __pe / __plink`, lượt TTA gốc).
- `research/probes/piece_object_probe.py`: F2 object từ mảnh (model hoặc `--oracle`).
- Chưa có: suy luận hai lượt trong app (`Predictor`): tầng 1 chọn nét, dựng mảnh của nét chọn + nét chạm, rồi tầng 2. Làm ở version sau nếu H141 đạt.

## 6. Dữ liệu, split, augmentation và metric

- e12 = e11 + mảnh, 203 trang, split hvac10b như E122 (`folds.json`). 3,25 triệu mảnh ứng viên, 528 nghìn mảnh ống; zip `datasets/openstroke_e12_data.zip`.
- Không trộn trang (`paste`), vì mảnh của nét bị đổi vùng không theo sang.
- Metric chính: F2 object theo mảnh (`piece_object_probe.py`, ngưỡng ống 0,5, ngưỡng cạnh 0,5, chốt trước), nhóm GT5; Viewer báo kèm (lớp Viewer
  không học, so lớp GT với quy ước Viewer có lệch). Phụ: F2 nét tầng 1 (log), số mảnh ống được kéo lên (tầng 1 p < 0,5 mà mảnh ≥ 0,5).
- Giới hạn: tập mảnh ứng viên khi chấm val dựng từ nhãn (nét true + nét chạm) nên lạc quan về R; số này đo khả năng dựng object, chưa phải pipeline
  đầu-cuối.

## 7. Baseline, ablation và recipe

| Run | Khởi tạo | Data | Cờ | Batch | Epoch |
|---|---|---|---|---|---|
| E122 (đối chứng tầng 1) | E114 | e11 | edge_pairs | 8 | 15 |
| **E127** | E122 `best.pt` | e12 | `piece_head=true` | 4 | 10 |

Mốc dựng object cùng giao thức ở §2 (gom theo lớp đúng 0,844–0,875; trần cạnh đúng 0,979).

## 8. Kỳ vọng và tiêu chí đạt / bác bỏ

Kỳ vọng (ước lượng, chưa đo): F2 object theo mảnh GT5 0,88–0,93, Viewer 0,75–0,85; F2 nét tầng 1 trong ±0,01 của E122.

Tiêu chí (ghi trước khi train, ngưỡng ống / cạnh 0,5):
- **Đạt:** F2 object theo mảnh GT5 ≥ 0,90 (vượt gom theo lớp đúng 0,875 dù lớp do model tự đoán), và F2 nét tầng 1 ≥ 0,840.
- **Bác bỏ:** GT5 < 0,85, hoặc F2 nét tầng 1 < 0,830.
- Ở giữa: vùng xám; xem lỗi theo lớp và theo loại cạnh trước khi đổi.

## 9. Kiểm thử và chi phí

`tests/test_os34.py` (6 test, CPU): cắt mảnh (cung, chữ T, độ dài đúng), gán object theo cạnh gần nhất, cạnh đối diện chỉ nối hai vách song song
chồng nhau, mảnh trong batch trỏ đúng nét cha sau augment / bỏ nét, đầu mảnh lan độ tin cậy chỉ qua cạnh (mảnh không nối không đổi), train tí
hon qua trainer thật lưu đầu ra mảnh vào `val_scores.npz`. Chi phí: mảnh ứng viên mỗi cửa sổ vài nghìn đến vài chục nghìn; đầu mảnh 64 chiều, 3
vòng; VRAM thêm ước lượng < 10 %; batch 4.

## 10. Lệnh train Colab

Upload `F:\Source_code\OpenStroke\datasets\openstroke_e12_data.zip` lên `MyDrive/Dataset/`.

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/os34 && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_e12_data.zip -d /content/os34
```

```python
!openstroke train \
    model=/content/drive/MyDrive/Train/OpenStroke31/e122_edge_pairs/fold0/best.pt \
    data=/content/os34/e12 \
    piece_head=true \
    batch=4 \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke34 \
    name=e127_pieces
```

Dòng `start` phải in `trains openstroke34 (new: piece_head)`; cột `piece_loss` có trong bảng loss. Gửi lại thư mục run (log, `fold0/val_scores.npz`).

## 11. Kết quả thật từ log

Chưa train.

## 12. Kết luận và version tiếp theo

Chưa có kết quả. Nếu H141 đạt: suy luận hai lượt trong app (tầng 1 → mảnh → tầng 2), object ra thẳng từ model; chấm rule_nl đầu-cuối so với hậu kỳ
hiện tại 0,835. Nếu bác bỏ: xem lỗi nằm ở lớp hay ở cạnh; nếu ở lớp, học lớp trên mảnh có ngữ cảnh rộng hơn (ảnh quanh mảnh).
