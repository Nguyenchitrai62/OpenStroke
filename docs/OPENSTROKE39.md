# OpenStroke39 — tiền huấn luyện tự giám sát trên kho PDF HVAC chưa gán nhãn (H156, E137)

User 2026-11-23 đồng ý tiền huấn luyện không nhãn trên kho PDF HVAC (`datasets/hvac_corpus`). Kiến trúc giữ nguyên openstroke28 (E114). Biến duy
nhất là **dữ liệu tiền huấn luyện**: model học cấu trúc bản vẽ HVAC bằng cách đoán lại đặc trưng của các nét bị che, trước khi học nhãn.

## 1. Bằng chứng

- Nút thắt là khả năng khái quát, không phải đơn vị token hay đầu ra: token theo mảnh chỉ nâng trần object +0,042 với 3,85 lần token (F215, không qua
  cổng AGENTS 0.12). Model tốt nhất E122 trên trang lạ còn cách trần 0,958 khoảng 0,15–0,26.
- Cùng kiến trúc: 19 trang GT, khởi tạo mới (E135a) đỉnh 0,837 ở epoch 3 rồi giảm (học thuộc sớm, F215); E114 (190 trang + tiền huấn luyện) 0,906.
- Tiền huấn luyện che nét là cải thiện duy nhất vượt nhiễu ở các lô trước: E19 +0,024 (FIRE chưa nhãn), E26 +0,04 F2 nét, E32 tiền huấn luyện có ảnh qua
  sàng lọc (F33, F36, F42). Lần đó kho chưa nhãn là **FIRE**; lần này là **HVAC** đúng miền bài toán.

## 2. Giả thuyết

**H156:** tiền huấn luyện che nét trên kho HVAC chưa nhãn (đúng miền, nhiều dự án hơn 10 PDF GT) giúp model nhận ra ống gió trên dự án lạ tốt hơn:
fine-tune cùng công thức E114 cho F2 nét Viewer8 (dự án khác) cao hơn E114.

## 3. Kiến trúc, dữ liệu, luồng

- Mạng: openstroke28 (U-Net transformer trên nét + nhánh ảnh ResNet18 + FPN, không layer). Không đổi.
- Tiền huấn luyện (E137p): `pretrain=0.4` = che 40 % nét mỗi cửa sổ (không vẽ chúng vào ảnh), học tái tạo đặc trưng của chúng (MSE). Không đọc nhãn.
- Dữ liệu tiền huấn luyện `datasets/u1` (`tools/make_pretrain_corpus.py`): 273 trang payload của kho HVAC (`tools/hvac_corpus.py`), bỏ trang thuộc PDF của
  trang val e12 và trang chung > 30 % đầu mút nét dài với trang val (chống rò). Không nhãn, không nhãn giả.
- Fine-tune (E137): khởi tạo `E137p/best.pt`, dữ liệu e12 split hvac10b (19 trang GT + 171 trang Viewer train, `viewer_share=0.65` như E114), 10 epoch,
  val 13 trang (GT5 + Viewer8), chọn best như E114.
- Nguồn SOTA: tiền huấn luyện che token kiểu MAE (He et al., CVPR 2022) / Point-MAE (Pang et al., ECCV 2022) cho đám điểm; ở đây token = nét, mục tiêu
  = đặc trưng nét. Lấy: che ngẫu nhiên + tái tạo. Bỏ: không tái tạo toạ độ điểm. Code có từ E19 (`pretrain=`).

## 4. Triển khai

- `tools/make_pretrain_corpus.py`: dựng u1 qua `openstroke.build.build_unlabeled(mode="hvac")`, chốt chống rò, `folds.json` (mọi PDF train, hai PDF làm
  fold val giả vì trainer cần, tiền huấn luyện không chấm điểm), zip.
- Không đổi code train / model (đường `pretrain=` có từ E19, chạy lại ở E26 / E32).

## 5. Baseline và ablation

| Run | Khởi tạo | Dữ liệu | Epoch |
|---|---|---|---|
| E114 (đối chứng) | tiền huấn luyện cũ (FIRE chưa nhãn / e32p) | e12 hvac10b, Viewer 0,65 | 10 |
| E137p | mới (ResNet18 ImageNet) | u1, che 40 % | 30 |
| **E137** | E137p | như E114 | 10 |

## 6. Kỳ vọng (ước lượng, chưa đo)

E137 so với E114 (chấm cặp, ngưỡng 0,20, TTA ×1): GT5 F2 nét ±0,01; Viewer8 +0,01 đến +0,03.

## 7. Tiêu chí (ghi trước GPU)

- **Đạt:** Viewer8 F2 nét E137 ≥ E114 + 0,02, **hoặc** GT5 ≥ E114 + 0,01 mà Viewer8 không thấp hơn.
- **Bác bỏ:** cả GT5 và Viewer8 < E114 + 0,005.
- Mốc E114 (chấm cặp F201): GT5 0,910, Viewer8 0,768.

## 8. Kiểm thử

Dựng u1 ở máy (chỉ đọc payload, không chạy model). Đường `pretrain=` đã có test từ E19 (`tests/`).

## 9. Lệnh Colab

Upload `datasets/openstroke_u1_data.zip` lên Drive. Cài code:

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/u1 && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_u1_data.zip -d /content/u1
```

E137p (tiền huấn luyện):

```python
!openstroke train \
    data=/content/u1/u1 \
    model=openstroke28 \
    epochs=30 \
    project=/content/drive/MyDrive/Train/OpenStroke39 \
    name=e137p_pretrain \
    pretrain=0.4 \
    tta=false
```

E137 (fine-tune, sau khi E137p xong):

```python
!openstroke train \
    data=/content/os34/e12 \
    model=/content/drive/MyDrive/Train/OpenStroke39/e137p_pretrain/fold0/best.pt \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke39 \
    name=e137_finetune \
    viewer_share=0.65
```

Chấm cặp E114 / E137:

```python
!python -m openstroke.compare_stroke_checkpoints /content/os34/e12 \
    /content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    /content/drive/MyDrive/Train/OpenStroke39/e137_finetune/fold0/best.pt \
    --baseline-label E114 --candidate-label E137 \
    --out /content/drive/MyDrive/Train/OpenStroke39
```

## 10. Kết quả thật

**E137** (log user 2026-11-23; tiền huấn luyện E137p bị dừng sớm ở epoch 6 / 30, fine-tune thừa hưởng `tta=False` nên chấm cuối TTA ×1). best.pt
epoch 4, ngưỡng 0,35: F2 nét GT5 0,899 (R 0,936 P 0,778), Viewer8 0,723, tất cả 0,805. Mốc E114 (chấm cặp, ngưỡng 0,20, TTA ×1): 0,910 / 0,768.
Viewer8 thấp hơn ~0,045; bắt thừa nặng vw_62b1 p3 (P 0,13), p2 (P 0,27). H156 **chưa chấm công bằng** (tiền huấn luyện mới 1/5); không có dấu
hiệu đột phá. Không ưu tiên chạy lại.

## 12. E138 / H157 — gom theo `layer_1` (màu + độ dày + loại nét) khi không có layer

User 2026-11-23 hỏi dùng `layer_1` (nhóm kiểu nét tự tạo như AI_Takeoff, `data.style_ids`). Hiện model không layer gom `LayerPool` trên **một
nhóm duy nhất** (mọi nét chung một "layer" rỗng), nên bước gộp theo nhóm gần như vô dụng; model chỉ thấy tỉ lệ nhóm kiểu nét qua hai đặc trưng
`style_*_share`. E31 (`pool_by=style`) từng bác bỏ nhưng **khi còn layer thật** (thay layer bằng nhóm kiểu nét làm mất thông tin); trường hợp không
layer chưa thử. Nhóm theo kiểu nét là quan hệ trong trang (nét cùng kiểu cư xử giống nhau), không phải giá trị tuyệt đối, giống cách logic HVAC
AI_Takeoff dùng `layer_1` khi trang không có layer.

- **H157:** fine-tune E114 với `pool_by=style` (cùng trọng số, `LayerPool` gom theo nhóm kiểu nét), mọi thứ khác giữ công thức E114. Viewer8 tăng
  vì nét ống cùng kiểu được kéo lên cùng nhau.
- Rủi ro: một nhóm kiểu bị gọi nhầm là ống thì bắt thừa cả nhóm (P giảm).
- **Đạt:** chấm cặp Viewer8 F2 nét ≥ E114 + 0,02 và GT5 không thấp hơn E114 − 0,005. **Bác bỏ:** Viewer8 < E114 + 0,005 hoặc GT5 < E114 − 0,01.

```python
!openstroke train     data=/content/os34/e12     model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt     epochs=10     project=/content/drive/MyDrive/Train/OpenStroke39     name=e138_layer1_pool     pool_by=style
```

```python
!python -m openstroke.compare_stroke_checkpoints /content/os34/e12     /content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt     /content/drive/MyDrive/Train/OpenStroke39/e138_layer1_pool/fold0/best.pt     --baseline-label E114 --candidate-label E138     --out /content/drive/MyDrive/Train/OpenStroke39
```

## 11. Điểm yếu và hướng sau

- 273 trang là ít hơn kho FIRE cũ (429); nếu H156 đạt, mở rộng kho HVAC (cùng chốt chống rò) là bước kế.
- Kho chưa nhãn có thể trùng dự án với trang train (không hại) — chốt chỉ chặn trang val.
