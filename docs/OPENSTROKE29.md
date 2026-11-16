# OpenStroke29 — model nét trả thêm lớp của từng nét (đầu lớp trên E114)

Ngày: 2026-11-16. Trạng thái: đã code, chưa train. Code: commit ghi ở §10 khi push.

## 1. Mục tiêu và phạm vi

Mục tiêu sản phẩm vẫn là F2 object ≥ 0,80 trên GT giữ ngoài train, với input không layer.

OpenStroke28 (E114) chỉ trả "nét nào cần bóc". Mọi việc còn lại (khối nào là object, object thuộc lớp nào) do hậu kỳ làm bằng luật hình học, và
gọi tên lớp là phần khó nhất. Version này thêm đúng một thứ: model trả xác suất lớp HVAC của từng nét. Hậu kỳ lấy lớp cho mỗi khối kín bằng
bỏ phiếu theo chiều dài các nét trên viền.

Giữ nguyên: dữ liệu và split (e8 / hvac10b), đầu nhị phân, kiến trúc thân (openstroke2 + nhánh ảnh FPN), cờ train của E114, hậu kỳ dựng khối
kín theo thứ tự của user (vùng kín trong item → vùng kín của nét chạm đầu → logic cho nét còn lại; F156–F159).

## 2. Bằng chứng và điểm yếu version trước

| Bằng chứng | Số | Nguồn |
|---|---|---|
| E114, 10 epoch, F2 nét val (13 trang) | 0,839 (R 0,886, P 0,693) | log user, F142 |
| E114 lại, 20 epoch | 0,839 (R 0,900, P 0,659); val chững từ epoch 13, loss train vẫn giảm | log user 2026-11-16 |
| Hậu kỳ thứ tự cũ (logic trước + vùng bù) | GT5 0,873, Viewer 0,589 | F151–F152 |
| Hậu kỳ thứ tự user + bộ phân loại hình dạng | GT5 0,698, Viewer 0,545 | F156–F159 |
| **Trần (oracle): cùng khối kín, lớp = phiếu theo lớp GT của nét viền** | **GT5 0,853, Viewer 0,778** | F160, `research/probes/oracle_class_probe.py` |

- Gấp đôi epoch không tăng F2 nét, nên model đã bão hòa ở đầu nhị phân. Thêm epoch không phải hướng đi.
- Với lớp đúng của từng nét, các lớp khó lên hẳn: côn thu P 0,79 / R 0,93, côn lệch tâm 0,85 / 0,94, Chân rẽ 0,88 / 0,92. Trước đó bộ phân
  loại hình dạng cho 0,37 / 0,81, 1,00 / 0,12, 0,31 / 0,58.
- Phần còn hở kể cả khi lớp đúng: thừa Duct (980 dự đoán / 644 GT). Đây là chuyện tách object, ngoài phạm vi version này.
- Trần là oracle, không phải số của model.

## 3. Giả thuyết đăng ký trước

**H130.** Một đầu lớp từng nét (CE trên nét đúng, đọc chung đặc trưng với đầu nhị phân) học được lớp HVAC đủ tốt để bỏ phiếu lớp cho khối kín
vượt bộ phân loại hình dạng.
- Biến duy nhất: thêm đầu lớp và loss lớp (khởi tạo từ E114, cùng data / split / cờ).
- Bác bỏ khi: object F2 theo thứ tự user với phiếu của model không vượt bộ phân loại hình dạng một khoảng rõ (xem §8), hoặc F2 nét tụt quá
  0,01 so với E114.

## 4. Kiến trúc và luồng dữ liệu

- Thân openstroke2 (PTv3 trên shape + nhánh ảnh ResNet18 FPN) như E114. Đặc trưng nét h (B, N, 64).
- Đầu nhị phân: giữ nguyên.
- **Đầu lớp** (`obj_head`, có từ E62): LayerNorm → Linear 64 → GELU → Linear → N_CLASSES + 2 (10 lớp `objects.HVAC_CLASSES` gồm Tee và
  các biến thể vuông tròn; 2 số offset tới tâm object, giữ nguyên như E62 nhưng hậu kỳ không dùng).
- Loss = BCE + Lovász (như E114) + 1,0 × (CE lớp, trọng số lớp theo căn chiều dài, chỉ trên nét đúng + smooth L1 offset).
- Suy luận: `Predictor.predict_payload` trả thêm `oprob` (n shape × 10, một góc nhìn); `openstroke scores` lưu `<stem>__oprob`; val cuối run
  lưu `__oprob` trong `val_scores.npz`.
- Hậu kỳ: `hvac_faces.post_objects(..., stroke_cls=oprob)` → `vote_names`: lớp khối = argmax tổng (xác suất lớp × chiều dài nét trên viền),
  cột 0 không bỏ phiếu. Không có xác suất lớp thì dùng bộ phân loại hình dạng như cũ.

**Tham chiếu SOTA vector / CAD.** Bài toán nhận dạng ký hiệu trên bản vẽ CAD vector (FloorPlanCAD: CADTransformer, SymPoint, SymPoint-V2,
VecFormer) đều có một đầu ngữ nghĩa theo primitive: mỗi primitive được gán lớp, rồi instance được ghép bằng truy vấn hoặc bỏ phiếu tâm.
- Lấy: đầu ngữ nghĩa theo primitive (ở đây primitive là shape của PDF).
- Bỏ: bộ giải mã instance bằng truy vấn. Instance ở đây lấy từ hình học khối kín, phần đã ổn định theo thứ tự user (F156–F159). Lý do: các thử
  nghiệm truy vấn / box trước đây (E69–E111) đều dưới 0,65.
- Điểm khác cho HVAC không layer: lớp gồm cả fitting ít mẫu (côn lệch tâm, Elbow vuông, Tee), nên dùng trọng số lớp. Không dùng PQ trên
  FloorPlanCAD làm kỳ vọng cho F2 HVAC.
- Chi phí: một MLP nhỏ, không đổi thân.

## 5. Cách triển khai

- Dataset: `tools/make_e9.py` → `datasets/e9` (e8 + `ocls`, `oinst`, `octr` bằng `objects.assign_objects`; object GT lấy từ gói chấm 203 trang
  `tools/make_score_pack_all.py`: AI_Takeoff cho trang doc (chỉ đọc), Viewer "Labels V1" cho trang vw). Zip `datasets/openstroke_e9_data.zip`.
- Train: cờ `obj_head=true` đã có (E62). Khởi tạo từ `best.pt` của E114: kiến trúc lấy từ checkpoint, đầu lớp khởi tạo mới (`load_weights`
  cho phép thiếu `obj_mlp.*`). Đầu nhị phân lúc bắt đầu cho đúng kết quả của E114 vì đầu lớp là nhánh riêng.
- Suy luận / app: `openstroke/infer.py` (oprob), `openstroke/score_all.py` (`__oprob`), `app/hvac_app.py` (truyền `stroke_cls`).
- Đánh giá local, không chạy model: `research/probes/oracle_class_probe.py` thêm chế độ đọc `__oprob` (§9).

## 6. Dữ liệu, split, augmentation và metric

- e9 = e8 (203 trang: 24 GT AI_Takeoff + 179 Viewer), split hvac10b giữ nguyên: val 5 GT + 8 Viewer, train 19 GT + 171 Viewer. Không thêm dữ
  liệu, không nhãn giả.
- Nhãn lớp: nét đúng (nhãn nhị phân của e8) được gán lớp của object chứa phần lớn chiều dài nó. Trang Viewer có nhãn "in_progress" (chưa gắn đủ).
- Augment như E114 (lật, xoay 90°, scale, màu, restyle); lớp đi theo nét nên không cần biến đổi riêng.
- Metric: F2 nét (trainer, như E114); độ đúng lớp trên nét đúng theo chiều dài (từ `val_scores.npz`); object F2 6 lớp, rule_nl, GT4 / GT5 /
  viewer8, như F151–F160.

## 7. Baseline, ablation và recipe

| Run | Khởi tạo | Biến đổi | Epoch | Ghi chú |
|---|---|---|---|---|
| E114 (đối chứng) | e32p | — | 10 / 20 | F2 nét 0,839 cả hai |
| **E116** | E114 `best.pt` | + đầu lớp (`obj_head=true`) | 10 | cùng data / split; công thức train đọc từ checkpoint E114 (lr0 0,002) |

Đối chứng hậu kỳ trên cùng val_scores: phiếu của model vs bộ phân loại hình dạng vs logic trước.

## 8. Kỳ vọng và tiêu chí đạt / bác bỏ (ghi trước train)

- F2 nét val: 0,83–0,85; không được dưới 0,828 (E114 − 0,01).
- Độ đúng lớp trên nét đúng (theo chiều dài, 6 lớp gộp): ≥ 0,85 trên GT5.
- Object F2 theo thứ tự user với phiếu của model: kỳ vọng GT5 0,78–0,83, Viewer 0,68–0,75 (trần 0,853 / 0,778).
- **Đạt:** Viewer ≥ 0,65 và GT5 ≥ 0,76.
- **Bác bỏ H130:** Viewer < 0,60 hoặc GT5 < 0,72, hoặc F2 nét < 0,828.
- Giữa hai mức: chưa rõ, xem lỗi theo lớp.
- Mốc sản phẩm 0,80 trên GT5 chỉ đạt được nếu sửa thêm việc tách Duct (F160).

## 9. Kiểm thử và chi phí

- `tests/test_hvac_faces.py`: `vote_names` (lớp của khối theo phiếu nét viền), `post_objects(stroke_cls=...)`.
- `tests/test_os28.py`: train ngắn với `obj_head=true`, `Predictor` trả `oprob` đúng kích thước, `openstroke scores` lưu `__oprob`.
- Chi phí: đầu lớp là một MLP nhỏ; thời gian một epoch kỳ vọng như E114 (~1:53 / 200 bước trên G4). Suy luận thêm một lượt đầu lớp (một góc
  nhìn): CPU app khoảng gấp đôi bước model.

## 10. Lệnh train Colab

Upload `F:\Source_code\OpenStroke\datasets\openstroke_e9_data.zip` (711 MB: e8 + lớp từng nét) lên `MyDrive/Dataset/`. Cần `best.pt`
của E114 (10 epoch) trên Drive. (Cách khác, không upload lại: giải nén e8 rồi chạy `openstroke add-classes` với
`datasets/openstroke_e9_objects.zip`, kết quả trùng khít.)

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/os29 && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_e9_data.zip -d /content/os29
```

```python
!openstroke train \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    data=/content/os29/e9 \
    obj_head=true \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke29 \
    name=e116_stroke_class
```

Kiến trúc (`openstroke-v1`) và công thức train của E114 (batch 4, lr0 0,002, `no_layers`, `viewer_share 0,65`, `restyle_mode order`,
`modes hvac`, `feature_drop layer_stats`…) được đọc từ chính checkpoint; đầu log in dòng `start` ghi rõ đã nạp gì. Chỉ `obj_head=true` là cờ
thí nghiệm. Cache đặc trưng mặc định nằm cạnh dataset (`/content/os29/cache`).

Sau run, chạy thêm (24 trang GT, có `__oprob`):

```python
!openstroke scores \
    model=/content/drive/MyDrive/Train/OpenStroke29/e116_stroke_class/fold0/best.pt \
    data=/content/os29/e9
```

Gửi lại: log, `fold0/val_scores.npz`, `fold0/all_scores.npz`, `fold0/best.pt`.

## 11. Kết quả thật từ log

Chưa train.

## 12. Kết luận và version tiếp theo

Chưa có. Nếu H130 đạt, việc kế là tách object Duct cho đúng quy ước GT (F160: thừa Duct kể cả khi lớp đúng).
