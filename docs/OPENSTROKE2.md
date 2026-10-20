# OpenStroke2: U-Net transformer tuần tự hoá (PTv3) trên shape, cửa sổ cả trang

Sổ cái: F8–F10 (bằng chứng), thẻ trước `docs/OPENSTROKE1.md`. Code: `openstroke/models/openstroke2.py`, `openstroke/data.py`,
`openstroke/serialize.py`, khung train `openstroke/engine/`. Test: `tests/test_v2.py`. Model: `model=openstroke2`.
Ghi ngày 2026-10-20, **trước khi train**.

## 1. Bằng chứng (đo local, CPU)

| Đo | Số | Nguồn |
|---|---|---|
| v1 trên Colab G4, fold 0 (H1_KS + 02.DN-TT-FP), 22 epoch đầu | val F2 **0,810 ở epoch 5**, rồi 0,809 / 0,784 / 0,788 trong khi loss train giảm 0,17 → 0,095: **overfit sớm**; precision 0,52–0,55 | log user 2026-10-20 |
| VRAM v1 (batch 8 × 4096) | 5 / 96 GB | user |
| Cửa sổ 4096 shape của v1 phủ đường chéo | trung vị **507 pt** (149–2587) | `research/probes/context_probe.log` |
| Mạng ống liền (nét true nối nhau trong 2 pt), đường chéo theo chiều dài | trung vị **856 pt**, FIRE 833–2369 pt | như trên |
| Chiều dài true nằm trong mạng dài hơn cửa sổ | 4096: trung vị **67 %** (FIRE 94–100 %); 16384: 26 %; **65536: 0 %** (trừ H1_KS 97 %, HVAC-B2 46 %) | như trên |
| Cửa sổ 65536 shape | trung vị 2433 pt ≈ cả trang (2739 pt) | như trên |
| Pooling lưới trên cửa sổ 65536 thật (H1_KS / 0875 / Y1 / 7.FP / 2D) | lưới (1, 2, 8, 32, 128) pt → 65536 / 6k–31k / 1,6k–5,5k / 270–810 / 31–99 token; lưới (1, 8, ...) nén 15–40× ngay bước đầu | đo trong session, `data._levels` |
| Trần chọn theo layer (E1.1) | F2 0,732; trang không layer 0,15–0,31 | sổ cái E1.1 |

Kết luận: v1 thiếu ngữ cảnh (đa số ống dài hơn tầm nhìn) và học thuộc (29 trang, không augment màu / tỉ lệ, không EMA).

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v2 (nguồn SOTA) | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Ống dài hơn cửa sổ 4096 (F8) | Cửa sổ **65536 shape** (≈ cả trang); U-Net 5 tầng pooling lưới 2 / 8 / 32 / 128 pt theo dịch mã z-order (**Point Transformer V3**) → tầng thô nhìn cả trang | F2 A ≥ B + 0,02 (B = cửa sổ 16384); FIRE lợi nhiều nhất | A ≤ B |
| kNN k = 16 chỉ nhìn 16 hàng xóm | Attention trong patch 1024 token liên tiếp của 4 đường cong (z, z-T, Hilbert, Hilbert-T), đổi đường cong theo khối và xáo khi train (**PTv3**), RoPE 2-D (**RoFormer / ViT-RoPE**) cho vị trí tương đối | – | – |
| Vị trí cục bộ | CPE: conv sâu 1-D (k = 7) dọc đường cong trước attention (thay cho sparse conv xCPE của PTv3) | – | – |
| Đặc trưng tay chỉ tả bbox | **ShapeEncoder**: PointNet trên 16 điểm lấy mẫu đều theo chiều dài cung của shape, chuẩn hoá theo cỡ (**VecFormer / CADTransformer**) + MLP 47 đặc trưng | – | – |
| Layer chia sẻ nhãn (E1.1, F7 OpenOBB) | LayerPool mean / max theo layer ở tầng 0 (mã hoá và giải mã), zero-init (**SymPoint-V2** LFE, OpenOBB H16) | – | – |
| Overfit sớm (F10) | drop path 0,3; EMA 0,999; augment lật / xoay 90° / tỉ lệ ±20 % / **hoán vị kênh màu** (màu là quy ước riêng từng văn phòng) / xám / bỏ ≤ 10 % shape; weight decay 0,05 | epoch tốt nhất ≥ 20 % số epoch | best ở ≤ 10 % số epoch |
| Metric theo chiều dài, ưu tiên recall | BCE (độ dài + 1)^0,5 × 2 cho true + **Lovász hinge** (PTv3 dùng CE + Lovász) | recall ≥ 0,90 | recall < 0,85 |
| Độ chính xác trước tốc độ | TTA 8 phép lật / xoay khi chấm `best.pt`; cửa sổ chồng 50 % lấy trung bình | TTA ≥ không TTA | – |

## 3. Chi tiết

- **Token** = shape có hình học: 47 đặc trưng f1 + 16 điểm (x, y) quanh tâm bbox (`features.shape_points`).
- **Dữ liệu** (`data.py`): trang xếp theo z-order tâm shape; mẫu train = 65536 shape liên tiếp, ngẫu nhiên; `collate` dựng sẵn bằng
  numpy cho từng cửa sổ: thứ tự z, 5 tầng (parent của từng ô, tâm trung bình), 4 hoán vị mỗi tầng. GPU chỉ gather / scatter.
- **Khối** (pre-norm): CPE → patch attention (RoPE, SDPA / flash) → LayerPool (tầng 0) → MLP 4×; drop path tăng tuyến tính 0 → 0,3.
- **Rộng / sâu:** dims (64, 96, 128, 192, 256), mã hoá (2, 2, 2, 4, 2), giải mã (1, 1, 1, 1), head dim 16. **5,65M tham số.**
- **Unpool:** gather đặc trưng tầng cha + chiếu skip (PTv3 SerializedUnpooling).
- **Zero-init:** LayerPool khởi tạo 0 (`test_v2_layerpool_starts_as_identity`). Bất biến padding (`test_v2_padding_does_not_change_outputs`).
- **Tận dụng phần có sẵn:** LayerPool, đặc trưng f1, metric, fold, E0 của v1; khung train lấy lại cơ chế OpenOBB (EMA mul/add, bàn console,
  results.csv / png, bỏ bước NaN, resume, patience). Phần mới: tuần tự hoá, U-Net, ShapeEncoder, Lovász, augment, TTA.

## 4. Bộ tham số đề xuất (mặc định của `model=openstroke2`) và lý do

| Tham số | Giá trị | Lý do |
|---|---|---|
| `window` | 65536 | F8: cửa sổ này mới chứa trọn mạng ống ở trang trung vị |
| `batch` | 4 | 262k shape / bước; chỉnh theo `openstroke bench` (chọn batch lớn nhất có peak < 80 GB) |
| `epochs` × `steps` | 100 × 200 = 20k bước | ~5 lần mỗi trang train / epoch ở batch 4; patience 30 dừng sớm nếu đã bão hoà |
| `lr0`, `lrf`, `cos_lr` | 2e-3, 0,01, cosine | AdamW cho transformer điểm (PTv3 6e-3 với batch 12 cảnh); nhỏ hơn vì 29 trang |
| `warmup_epochs` | 3 | transformer cần warmup; 600 bước |
| `weight_decay` | 0,05 | PTv3 / ViT; không áp lên norm / bias |
| `drop_path` | 0,3 | PTv3 dùng 0,3; chống overfit F10 |
| `ema` | 0,999 | YOLO 0,9999 cho ~100k bước; 20k bước → 0,999 |
| `clip` | 1,0 | ổn định với Lovász |
| `pos_weight`, `alpha`, `lovasz` | 2, 0,5, 1 | ưu tiên recall; metric theo chiều dài; Lovász tối ưu IoU trực tiếp |
| augment | flip 0,5, rot90, scale 0,2, color 0,5, gray 0,1, drop_tokens 0,1 | F10; màu / độ dày là quy ước riêng từng bản vẽ |
| `tta` | true | độ chính xác trước tốc độ |

## 5. Bảng chỉ số kỳ vọng (ghi trước)

Hai hạt giống khác **đúng một yếu tố** (cửa sổ): **A** `model=openstroke2` (window 65536), **B** `model=openstroke2 window=16384`.
Cùng fold, cùng bỏ 5 trang trùng, cùng mọi tham số khác. Mốc ngoài: v1 cùng fold (lần chạy của user) và GBDT.

| Chỉ số (micro theo chiều dài, mọi fold val, ngưỡng của fold) | Kỳ vọng | Bác bỏ nếu |
|---|---|---|
| F2 A − F2 v1 | ≥ +0,05 | < +0,02 |
| F2 A − F2 B | ≥ +0,02 (FIRE ≥ +0,03) | A ≤ B |
| Recall A | ≥ 0,90 | < 0,85 |
| Trang val không đánh dấu QA có F2 < 0,5 | 0 | ≥ 2 |
| Epoch tốt nhất / số epoch đã chạy | ≥ 0,2 | ≤ 0,1 (vẫn overfit) |
| F2 có TTA − không TTA (cùng best.pt) | ≥ 0 | < −0,005 |

## 6. Lệnh (Colab G4)

```
!openstroke bench model=openstroke2 batch=2,4,8 window=65536
!openstroke train data=/content/dataset_local/e0 model=openstroke2 epochs=100
!openstroke train data=/content/dataset_local/e0 model=openstroke2 epochs=100 window=16384 batch=16
```

B dùng batch 16 để số shape mỗi bước bằng A (262k). Gửi lại: zip của ô 5 (có `train.log`, `summary.json`, `results.csv` từng fold).

## 7. Kết quả thật

(chờ log)
