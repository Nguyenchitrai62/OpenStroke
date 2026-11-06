# OpenStroke7: model HVAC ra thẳng object

Thẻ version (AGENTS.md mục 7). Ghi ngày 2026-11-06, **trước khi train**. Thẻ trước: `docs/OPENSTROKE6.md`.

**Yêu cầu (user, 2026-11-06):** tách theo hệ; với HVAC, model ra thẳng object và nét nào (lệnh vẽ nào) thuộc object nào, như JSON cuối của AI_Takeoff. Chỉ cần các loại logic hiện dùng: Duct, Elbow, Tee, Concentric / Eccentric Reducer, Chân rẽ, Elbow vuông, Chân rẽ / Reducer vuông tròn. Ngân sách: sàn 550 CU, tức 50 CU cho phần này.

## 1. Bằng chứng

Số trên 19 trang HVAC GT có lần chạy pipeline còn lưu (không tính `Y1_10-20.pdf`), `research/probes/objects_oracle_probe.py`.

| Đo | Số |
|---|---|
| Nét ống `true` gán được object GT (đa giác GT nhỏ nhất chứa phần lớn chiều dài nét) | 100 % chiều dài |
| Object dựng lại từ lớp đúng và tâm đúng | F1 0,997 |
| Như trên, tâm lệch ±5 pt | F1 0,690 (P 0,553, R 0,916) |
| Như trên, tâm lệch ±15 pt | F1 0,448 |
| Pipeline AI_Takeoff hiện tại, cùng cách khớp | F1 0,871 (P 0,822, R 0,925) |

Số object GT: Duct 2.830, Elbow 1.099, Concentric Reducer 446, Chân rẽ 391, Chân rẽ vuông tròn 244, Eccentric Reducer 236, Elbow vuông 33, Reducer vuông tròn 16. **Tee: 11 object trong cả GT, không đủ để học; model có lớp Tee nhưng không kỳ vọng gì ở nó.**

Khớp object: object dự đoán khớp object GT cùng lớp khi chiều dài nét chung ≥ 50 % hợp của hai bên, mỗi object khớp một lần.

## 1b. Phân tích trước khi train (user 2026-11-06: input là ảnh + JSON vector, output là JSON cuối)

JSON cuối HVAC của AI_Takeoff (`normalized.json`, 37 job, 16.809 object primary) có cho mỗi object: `shape_name`, `vertices`, `dn` (kích thước, 79 % object có, ví dụ `1400x400`), `connectors` / `pipe_id` (nối mạng), `system_type`, `count_base` (theo chiều dài hay theo cái), `layer`; cộng tổng khối lượng theo trang. Payload vector hiện không chứa chữ.

| Phần của JSON cuối | AI_Takeoff đang làm bằng | Cách đề xuất |
|---|---|---|
| Lớp, đa giác, nét thuộc object | chọn layer + logic hình học | model (E62) |
| `dn` | OCR chữ `AxB` + tỷ lệ bản vẽ + topology (`hvac_dimensions.infer_hvac_dimensions`) | giữ logic đó, đưa object của model vào; chỉ học từ chữ nếu logic không đạt |
| `connectors`, `pipe_id` | topology của SDK | suy từ object kề nhau |
| `system_type`, `count_base` | theo lớp | quy tắc |

Mức pipeline trên 19 trang (`research/probes/objects_dn_probe.py`): object khớp có cùng `dn` với GT 0,843 (Duct 0,902, Elbow 0,916, Chân rẽ 0,654, Chân rẽ vuông tròn 0,025: có thể chỉ khác cách ghi, chưa kiểm). Tổng sai khối lượng theo kích thước so với GT: Duct (chiều dài) 0,136; Elbow (số cái) 0,258; Reducer 0,49–0,56; Chân rẽ 0,58.

Không sinh thẳng JSON từ ảnh (kiểu DETR hay model sinh văn bản): mỗi trang 300–1.000 object, chỉ ~30 trang GT, và cách đó không trả được "lệnh vẽ nào thuộc object nào". Ảnh thật của trang chỉ cần cho chữ (OCR) và PDF scan; PDF scan không có vector nằm ngoài hướng này.

## 2. Cơ chế

Theo hướng panoptic của CADTransformer / SymPoint (mỗi phần tử có lớp và véc-tơ tới tâm đối tượng), trên model nét hiện có:

1. **Đầu ra mới `obj_head`**, đứng riêng, không đổi đầu `true` / `other`: mỗi nét cho xác suất 10 lớp (`objects.HVAC_CLASSES`) và véc-tơ tới tâm object (đơn vị 100 pt). Test: logit ống giữ nguyên khi thêm đầu này.
2. **Loss:** cross-entropy lớp (trọng số nghịch căn bậc hai chiều dài từng lớp) cộng smooth L1 véc-tơ tâm, chỉ trên nét có object, trọng số theo chiều dài như loss chính.
3. **Nhãn:** dataset `datasets/e7` = `e6` thêm `ocls`, `oinst`, `octr` (`openstroke build-objects`). Trang nhãn giả HVAC lấy object của pipeline, như nhãn nét.
4. **Dựng object:** nét được hậu kỳ giữ lại → lớp có xác suất cao nhất → các nét cùng lớp có tâm cách nhau dưới 6 pt là một object. Đa giác: hình chữ nhật xoay nhỏ nhất (Duct, Reducer, Chân rẽ) hoặc bao lồi (Elbow). Đầu ra: danh sách object, mỗi object có lớp, đa giác và chỉ số các lệnh vẽ.
5. **Augment:** tâm object lật, xoay, co giãn cùng nét.

## 3. Giới hạn đã biết

- Gom theo tâm rất nhạy với sai số: ±5 pt đã tụt F1 từ 0,997 xuống 0,690. Nếu model cho tâm sai nhiều, phải đổi cách gom (gom theo nét chạm nhau cùng lớp, hoặc dự đoán thêm kích thước object) trước khi kết luận model hỏng.
- Pipeline đạt 0,871 trên GT mà 71 % object sinh từ chính pipeline (F29): số pipeline lạc quan.
- Chưa có kích thước (`dn`, ví dụ `150x150`) hay hướng dòng chảy: chỉ lớp, đa giác và nét.
- 10 trang `Y1_10-20.pdf` cũ mất GT (user thay bằng bản mới, doc214); bản mới và `UAT_clean_B2_1-200_v1.pdf` (doc211, 315 object) chưa vào dataset.

## 3b. Hướng B: OpenOBB riêng cho HVAC (E63, user 2026-11-06)

User chọn train lại OpenOBB với data tạo từ GT HVAC: model ảnh không cần layer nên tổng quát hơn pipeline (pipeline sập khi không có layer, F55), chấp nhận thấp hơn một chút.

- **Ảnh:** vẽ lại từ chính vector của payload (màu, độ dày, mảng tô thật; không PyMuPDF, không layer), 4 px / pt để Chân rẽ (cạnh ngắn 2,3 pt) còn khoảng 9 px.
- **Nhãn:** hình chữ nhật xoay nhỏ nhất của đa giác GT; 9 lớp (`objects.HVAC_CLASSES` trừ `other`).
- **Train:** `openobb train` với `tile=True tile_scale=1` (tile 1024 px chồng 200 px như SAHI), tinh chỉnh từ `openobb24x.pt`.
- **Gắn về nét:** hộp của model → nét theo đúng quy tắc nhãn v3b → mỗi object có lớp, hình, điểm tin cậy và danh sách lệnh vẽ.
- Tiêu chí: `research/HYPOTHESES.md` H78. Mốc pipeline trên 7 trang val: F1 object 0,858.

## 4. Kỳ vọng ghi trước

Tiêu chí ở `research/HYPOTHESES.md` H77.

## 5. Điều chứng minh sai

- F1 object dưới 0,70 trên trang val dù F2 nét không giảm: model không học được tâm, đổi cách gom trước khi đổi model.
- Đúng lớp nét nhưng object sai: lỗi ở cách gom.
- F2 nét HVAC giảm hơn 0,0125 so với E37 cùng trang: đầu mới làm hại đầu cũ.

## 6. Lệnh

```bash
openstroke build-objects datasets/e6 datasets/e7
```

```bash
openstroke train model=runs/e37 data=/content/dataset_local/e7 folds=/content/dataset_local/e7/folds_cv_pseudo.json fold=0,1,2,3,4 modes=hvac feature_drop=layer_stats batch=4 lr0=0.002 image_branch=true img_lum=true restyle_mode=order layer_head=true img_fpn=true obj_head=true epochs=12 name=e62
```

`model=runs/e37`: mỗi fold tinh chỉnh từ `best.pt` E37 của chính fold đó, không rò val.

## 7. Kết quả thật

Chưa có.
