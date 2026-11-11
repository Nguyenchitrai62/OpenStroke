# OpenStroke13: hạt giống từ hộp detector + đồ thị nét (HVAC không layer)

Ngày 2026-11-11. Trạng thái: **E92 và E93 có code + test, chưa chạy GPU; E94 chưa có code**. Phạm vi như v10–v12: PDF vector HVAC không layer, ảnh vẽ từ
vector + token nét, ra object và danh sách shape của mỗi object. Không đổi dữ liệu, fold, GT, protocol chấm (`dataset_vectors.sampled8.v1`
của F93). Repo OpenOBB và AI_Takeoff chỉ đọc.

## 1. Vì sao v10–v12 ra F2 object ~0,1 (F95)

**Log v12 `e91_f2_native2`** (user, lr 0,001, batch 32, từ E85): F2 object đồ thị epoch 1–5 là 0,105 / 0,059 / 0,092 / 0,079 / 0,078,
recall 0,055–0,108, precision 0,072–0,096; tốt nhất ở epoch 1 rồi đi ngang. Dừng ở epoch 6. Log v10 / v11 đã cho thấy hai đầu của đồ thị
yếu: cạnh "cùng object" P 0,025–0,07 ở R 0,57–0,81 (F83, F85, F86); lớp đúng trên nét thuộc object chỉ 0,12–0,18 (F83, F85).

**Cơ chế trong mã** (`openstroke/obb/models/stroke_graph.py`, `predict.graph_objects`), cả bốn cùng đẩy về một hướng:

1. **Loss cạnh cân bằng** (trung bình dương + trung bình âm, `graph_loss`): xác suất cạnh được học như prior 50 %, trong khi trên tập
   ứng viên k-NN của 7 trang chỉ **4.322 / 2.717.742 cạnh = 0,16 %** là cùng object (`research/probes/graph_error_sim.py`).
2. **Ngưỡng cạnh cố định 0,5** khi giải mã (`edge_thr=0.5`, `keep_thr=0.5`), không bao giờ chỉnh trên val; chọn best chỉ dò ngưỡng
   độ tin cậy object 0,1–0,5. Với prior lệch 600 lần, ngưỡng 0,5 nhận rất nhiều cạnh sai.
3. **Gộp ô bằng giá trị lớn nhất** (`GraphPage.add`): một ô báo nhầm là cạnh sai được giữ.
4. **Thành phần liên thông + lớp lấy từ nét**: một cạnh sai nối hai object; lớp object = trung bình lớp nét theo chiều dài, mà lớp nét chỉ
   đúng 12–18 %. Phân lớp ở mức nét bỏ qua thông tin cấp object mà detector dùng (detector chỉ sai lớp 5–9 %, F78).

v12 (một xác suất HVAC + soft-F2) chỉ đổi xác suất nền / HVAC, không chạm bốn điểm trên, nên không có lý do để tăng. Run của user còn dùng
**lr 0,001** (thẻ v12 ghi 1e-4): tinh chỉnh model đã hội tụ ở lr này làm tụt detector (F79, E83 epoch 1 mAP50 0,294).

**Mô phỏng 0 GPU** (`graph_error_sim.py`): lấy nét và object GT của 7 trang, làm hỏng cạnh và lớp đúng như mức trong log, giải mã như
`GraphPage`. Cổng nét giả định hoàn hảo (chỉ giữ nét thuộc object GT), cạnh sai được rải giữa các object GT cạnh nhau.

| Cạnh (R / P) | Lớp nét đúng 0,18 | 0,60 | 1,00 |
|---|---|---|---|
| 0,60 / 0,03 (mức log v10) | **0,052** | 0,183 | 0,307 |
| 0,80 / 0,07 (mức log v11) | **0,053** | 0,182 | 0,306 |
| 0,80 / 0,50 | 0,069 | 0,202 | 0,348 |
| 0,85 / 0,80 | 0,128 | 0,405 | 0,651 |
| 0,90 / 0,95 | 0,159 | 0,543 | 0,864 |
| 1,00 / 1,00 | 0,182 | 0,604 | 0,946 |

Ở mức chất lượng trong log, mô phỏng ra F2 ~0,05, khớp mức 0,06–0,10 của v12. **Lớp nét là chỗ chặn lớn nhất**: cạnh hoàn hảo mà lớp nét
đúng 18 % vẫn chỉ 0,18. Muốn ≥ 0,85 cần cả cạnh P ≥ 0,95 / R ≥ 0,9 lẫn lớp gần đúng hết.

**Hạt giống từ hộp** (cùng mô phỏng): cạnh chỉ được tính khi hai nét cùng thuộc một hộp E69 (conf ≥ 0,2), lớp object lấy từ hộp.

| Cạnh (R / P) | F2 (P / R) |
|---|---|
| 0,60 / 0,03 | **0,524** (0,57 / 0,51) |
| 0,80 / 0,07 | 0,534 (0,61 / 0,52) |
| 0,80 / 0,50 | 0,538 (0,61 / 0,52) |
| 1,00 / 1,00 | 0,624 (0,66 / 0,61) |

Với đúng các cạnh kém hiện tại, gieo hạt từ hộp kéo F2 từ ~0,05 lên ~0,52, ngang E69 nét model tự trả (0,513). Trần của cách này bị chặn
bởi recall của detector (0,624 với cạnh hoàn hảo). Hai số đo trong bảng là mô phỏng trên protocol score pack, chưa phải kết quả model.

## 2. Cơ chế OpenStroke13

Giữ backbone, nhánh vector, detector, đầu membership và đầu cạnh của v10–v12. Đổi **cách dựng object** và **cách học cạnh**:

1. **Hạt giống = hộp detector.** Mỗi hộp (lớp, điểm, nét `members` của nó) là một object ứng viên. Lớp object lấy từ hộp, không từ lớp nét.
2. **Đồ thị chỉ sửa membership quanh hạt giống.** Từ `members` của hộp, lan theo cạnh có xác suất ≥ τ_e, nhưng chỉ tới nét nằm trong đa giác
   hộp nới thêm `m` pt. Lan được ra ngoài hộp một chút (ống bị cắt giữa hai ô), nhưng không thể nối tràn qua cả trang. Nét bị nhiều hạt giống
   giành thì về hạt giống điểm cao nhất (như `members`).
3. **Học cạnh đúng tỉ lệ thật.** Bỏ trung bình cân bằng; dùng BCE có `pos_weight` vừa phải (đề xuất 20), cạnh ứng viên chỉ giữa nét được
   cổng HVAC giữ. Gộp ô bằng trung bình thay cho lớn nhất. τ_e, `m` và ngưỡng độ tin cậy chọn trên val (cùng lưới, một lượt suy luận).
4. **Object không có hạt giống** (để vượt recall của detector) chỉ bật ở bước sau, khi cạnh đạt P ≥ 0,8 ở R ≥ 0,8 trên val: thành phần liên
   thông không chạm hộp nào, đủ dài, xác suất HVAC cao, lớp từ đầu lớp cấp object (`object_class` của v11) chứ không từ lớp nét.
5. **Mỗi epoch chấm ba đường** trên cùng protocol, cùng một lượt suy luận: `seeded` (v13), `members` (hộp + nét model tự trả, mốc ngay trong
   run), `graph` (đồ thị cũ). Best theo F2 của `seeded`. Không còn tình trạng thiếu mốc native như run v12.

Hậu kỳ vẫn tối thiểu: lan theo cạnh trong vùng hộp + gán nét theo điểm. Không layer, không logic HVAC theo layer.

## 3. Thí nghiệm và kỳ vọng ghi trước

Mọi run: từ E85 `e85_graph2/weights/best.pt` (hoặc E83 nếu cần so không đồ thị), lr **1e-4**, warmup 0,5, batch 32 / nbs 64, 10 epoch,
seed 0, cap 8192, scale 1, protocol native, 7 trang. Một biến mỗi run. Chi tiết H107–H109 ở `research/HYPOTHESES.md`.

| Run | Biến duy nhất | Đạt | Bác bỏ |
|---|---|---|---|
| E92 / H107 | Chỉ đổi giải mã sang `seeded` trên checkpoint có sẵn (E85, E91), **không train** | F2 `seeded` ≥ F2 `members` cùng checkpoint + 0,02 | F2 `seeded` ≤ F2 `members` |
| E93 / H108 | Học cạnh theo tỉ lệ thật (`graph_edge_pos_weight=20`) + gộp ô trung bình (`graph_pool=mean`), so với đối chứng E93c cùng lịch | P cạnh ≥ 0,5 ở R ≥ 0,8 trên val và F2 `seeded` +0,02 so với E92 | F2 `seeded` không tăng |
| E94 / H109 | E93 + object không hạt giống | R object +0,03 và F2 không giảm so với E93 | R không tăng hoặc F2 giảm > 0,01 |

E92 rẻ nhất (chỉ chấm, vài phút GPU) và quyết định có đi tiếp không. E94 chỉ chạy khi E93 đạt điều kiện cạnh.

## 4. Điều chứng minh sai

- **E92 bác bỏ** (`seeded` ≤ `members`): đồ thị không thêm gì vào hộp; dừng nhánh đồ thị, tập trung recall detector (ô nhiều tỉ lệ, bỏ cap
  8.192 nét / ô, dữ liệu).
- **E93 không lên được P cạnh ≥ 0,5**: đặc trưng nút / hình học chưa đủ để phân biệt cặp nét cùng object; không bật E94.
- Mô phỏng giả định cổng hoàn hảo; cổng thật (R ~0,87 ở v11) sẽ thấp hơn. Trần `seeded` bị chặn bởi recall detector (~0,52–0,62).

## 5. Phạm vi kỳ vọng

v13 nhắm **sửa sụp đổ** (0,1 → ngang hoặc hơn đường hộp) và tạo đường đo rõ cho cạnh. Nó chưa phải bước tới mức thay pipeline có layer
(F2 0,888): phần còn lại vẫn là recall của detector (Duct sót 33 %, F82) và dữ liệu. Không tăng kỳ vọng sau khi có số.

## 6. Lệnh

**E92 (code đã có, `openstroke/obb/seeded.py`):** chỉ chấm, không train. Một lượt suy luận cho ra cả hộp, đồ thị và object hạt giống với
ba ngưỡng cạnh (0,5 / 0,7 / 0,9, vì đầu cạnh chưa hiệu chỉnh theo tỉ lệ 0,16 %); bộ chấm native in F2 / R / P của `members`, `graph`,
`seeded@0.5`, `seeded@0.7`, `seeded@0.9`, kết quả chính là `seeded@t` tốt nhất. Lề vùng hộp 16 px (4 pt ở 4 px / pt).

```
!openstroke obb val   model=/content/drive/MyDrive/Train/OpenStroke10/e85_graph2/weights/best.pt   data=/content/openstroke9_e83/obb_hvac7/data.yaml   graph_decode=seeded
```

Lặp lại với `model=.../OpenStroke12/e91_f2_native2/weights/best.pt`. Mỗi lệnh vài phút GPU.

**E93 (code đã có):** train từ E85, lr **1e-4** (không phải 1e-3), các cờ v11 / v12 tắt để chỉ đổi đầu cạnh. `graph_decode=seeded`:
mỗi epoch chấm `members`, `graph`, `seeded@0.3/0.5/0.7/0.9`, best theo `seeded` tốt nhất, log thêm một dòng các đường. Hai run, cùng lịch:

| Run | `graph_edge_pos_weight` | `graph_pool` | `name` |
|---|---|---|---|
| E93c (đối chứng) | 0 (loss cân bằng cũ) | max | `e93c_control` |
| E93 | 20 | mean | `e93_edges` |

```
!openstroke obb train   data=/content/openstroke9_e83/obb_hvac7/data.yaml   model=/content/drive/MyDrive/Train/OpenStroke10/e85_graph2/weights/best.pt   epochs=10   batch=32   lr0=0.0001   warmup_epochs=0.5   nbs=64   seed=0   max_iters=190   stroke_graph=True   endpoint_members=True   graph_foreground=False   graph_hard_negative=False   graph_object_class=False   graph_unified=False   graph_f2=False   graph_decode=seeded   graph_edge_pos_weight=20   graph_pool=mean   select_metric=object_f2   imgsz=1024   tile=True   tile_scale=1   max_tokens=8192   project=/content/drive/MyDrive/Train/OpenStroke13   name=e93_edges
```

Đối chứng E93c: cùng lệnh, đổi ba dòng thành `graph_edge_pos_weight=0`, `graph_pool=max`, `name=e93c_control`.
Đọc dòng `Epoch ... | object F2` (đường `seeded` tốt nhất) và dòng thụt lề ngay dưới (cả 6 đường). Phần "cạnh sau cổng" trong cơ chế
mục 2 chưa đưa vào E93 để giữ đúng hai thay đổi; ứng viên cạnh vẫn là k-NN trên mọi nét như v10.

## 7. Kết quả thật

(trống: lấy từ log train / val)
