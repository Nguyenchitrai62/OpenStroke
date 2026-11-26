# OpenStroke

Thư viện phân loại nét vector từ lệnh vẽ PDF bản vẽ CAD. Đầu vào là **lệnh vẽ vector của một trang** (`split_countfire` JSON: hình học, màu, độ dày, nét đứt, fill); đầu ra là **kết quả raw: xác suất từng shape** (kèm xác suất cạnh cùng object và điểm cụm nếu checkpoint có đầu tương ứng).

**Thiết kế tách biệt cho triển khai dịch vụ (giống OpenOBB):**
- **Không cần truyền hệ (`mode`) khi suy luận:** model đọc thuần lệnh vẽ vector; loại nét cần bóc hoàn toàn do dữ liệu huấn luyện quyết định (HVAC, FIRE, PLUMBING hay bộ dữ liệu tuỳ biến đều dùng chung một API).
- **Trả kết quả raw, hậu kỳ tách thành module riêng:** `OpenStroke.predict(...)` chỉ nạp model, suy luận và trả kết quả raw (không import hay chạy bất kỳ logic hậu kỳ nào), thuận tiện host độc lập từng model thành microservice.
- Không dùng Ultralytics; không import PyMuPDF (AGPL). Kiến trúc mới nhất: **`openstroke46`** — **một kiến trúc vector+ảnh phổ quát duy nhất** (`PTv3 U-Net + CADStemV2 2048px + 4-layer edge_style GAT + 18D graph_topo_uni`) với **đầu ra động theo dữ liệu huấn luyện** (tự nhận nhị phân hoặc `K` lớp bất kỳ + `class_names` từ `manifest.json`, hỗ trợ chuyển giao trọng số xuyên hệ). Phiên bản **`0.48.0`**.

---

## Cài đặt

**Máy có GPU NVIDIA:**

```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -U "openstroke @ git+https://git.anybim.vn/CxDP/worker/modules/openanything/hicasstroke.git"
```

**Máy chỉ có CPU:**

```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -U "openstroke @ git+https://git.anybim.vn/CxDP/worker/modules/openanything/hicasstroke.git"
```

- Repo GitLab nội bộ cần token: `git+https://<user>:<token>@git.anybim.vn/CxDP/worker/modules/openanything/hicasstroke.git`.
- Python ≥ 3.10. GPU RTX 50xx: đổi `cu126` thành `cu130`.
- Tuỳ chọn: `openstroke[app]` (app thử model), `openstroke[dev]` (pytest), `openstroke[baseline]` (`scikit-learn` cho hậu kỳ độc lập).

---

## Suy luận & Triển khai (Giao diện kiểu OpenOBB)

### Python API

```python
from openstroke import OpenStroke

# 1. Nạp model một lần lúc khởi động service (tự động GPU bf16 + gộp Conv/BN + channels_last, hoặc CPU)
model = OpenStroke("models/best_deploy.pt")
model.warmup()                                  # làm ấm CUDA / thread pool một lần lúc boot

# 2. Suy luận raw — KHÔNG cần truyền mode (nhận dict payload, file .json / .json.gz, thư mục, glob, hoặc list)
results = model.predict(payload)                # hoặc gọi tắt: results = model(payload)

for r in results:
    r.prob          # np.float32 (N,): xác suất RAW từng shape theo đúng thứ tự payload["shapes"] (với model K lớp = 1 - p_other)
    r.cls_prob      # np.float32 (N, K) | None — xác suất từng lớp raw (động theo dữ liệu huấn luyện, vd. [p_other, p_main, p_branch])
    r.class_names   # tuple[str, ...] | None — tên lớp tương ứng từ checkpoint (vd. ("other", "main", "branch"))
    r.valid         # np.bool_   (N,): mặt nạ shape có hình học được model chấm (shape rỗng = False, prob = 0)
    r.threshold     # float: ngưỡng tham khảo từ val (KHÔNG tự cắt vào r.prob)
    r.edges         # (pairs (E, 2), prob (E,)) | None — xác suất cặp nét nối thuộc cùng object (nếu có edge_head)
    r.groups        # dict | None — cụm & xác suất 8 lớp (nếu model có group_head)
    r.speed         # {"preprocess": ms, "inference": ms, "postprocess": ms}
    raw_resp = r.to_dict(compact=True)          # dict nén thưa (~10-20x nhỏ hơn trên trang A0) trả qua HTTP / RPC
```

**Phía Caller (module hậu kỳ tách riêng nhận `raw_resp` từ API model):**

```python
from openstroke import Results
from openstroke.hvac_faces import postprocess_hvac       # hoặc: from openstroke.line_decoder import postprocess_lines

raw = Results.from_dict(raw_resp)                        # bung lại đầy đủ prob / valid / edges / groups
hvac = postprocess_hvac(payload, raw, classifier_path="models/hvac_faces.joblib")
# FIRE / PLUMBING: pipes = postprocess_lines(payload, raw, mode="fire")
```

- **Chế độ nhanh chỉ lấy xác suất nét:** truyền `include_groups=False` (hoặc thêm `include_edges=False`) khi module hậu kỳ chỉ cần `prob`.
- **Xử lý thư mục lớn tiết kiệm RAM:** dùng `stream=True` (`for r in model.predict("pages/", stream=True): ...`).
- **Tối ưu tốc độ deploy:**
  - Tự động **gộp `Conv2d + FrozenBatchNorm2d`** trong `ImageBranch`, bật `channels_last` + TF32 + `cudnn.benchmark` + AMP bf16/fp16 trên GPU (`flush_denormal` trên CPU).
  - Vẽ raster chung 1 lượt (`render_multi`) cho cả 3 kênh CAD (`openstroke44`/`45`), song song hóa tiền xử lý (`tokens + shape_points + graph`), loại đồng bộ CUDA thừa khi `no_layers=True`.
  - Hỗ trợ `compile=True` (hoặc `OPENSTROKE_COMPILE=1`) cho server chạy lâu trên GPU (mặc định tắt, như OpenOBB).
- **Tương thích ngược:** `openstroke.infer.Predictor` vẫn được giữ nguyên (`mode` giờ là tham số tuỳ chọn, mặc định `None`).

### CLI kiểu YOLO / OpenOBB

```bash
# Suy luận raw từ file, thư mục hoặc glob
openstroke predict model=best_deploy.pt source=pages/ save=true

# Cắt gọn checkpoint để giao đi / deploy (chỉ giữ trọng số EMA + cấu hình mạng, bỏ tham số train; half=true giảm ~50% dung lượng)
openstroke strip model=runs/e150/fold0/best.pt half=true

# Đo thời gian xử lý thực tế (preprocess / inference / raw-head)
openstroke profile model=best_deploy.pt source=page-0001.v0.json.gz
```

---

## Quy chuẩn đầu vào & Checkpoint

### Đầu vào: payload một trang

JSON định dạng `split_countfire` (`.json`, `.json.gz` hoặc `dict`). Model **chỉ đọc `payload["shapes"]`**:

```python
payload = {
    "shapes": [
        {"items": [["l", [x0, y0], [x1, y1]], ["c", p0, p1, p2, p3]],     # l đoạn thẳng, c bezier bậc 3, re chữ nhật, qu tứ giác
         "type": "s", "color": [0, 0, 0], "fill": None, "width": 0.35,
         "dashes": "[] 0", "closePath": False, "seqno": 1532, "layer": "M-DUCT"},
    ],
}
```

- Toạ độ: pt, hệ display top-left (đã xoay theo `/Rotate`).
- Một shape = một đơn vị nhãn; không tách hay gộp shape trước khi đưa vào.
- Trường `layer` không được đọc ở các kiến trúc không layer (`no_layers=True`, từ `openstroke28` trở đi).

### Checkpoint

Một model = một file `.pt` (dùng thẳng `best.pt` hoặc bản cắt gọn `best_deploy.pt` từ `openstroke strip`). File tự mang kiến trúc (`config["arch"]`), trọng số EMA, danh sách đặc trưng `feature_names`, cỡ cửa sổ `window`, ngưỡng tham khảo `threshold` và `tta`. Nếu code lệch bộ đặc trưng sẽ báo lỗi ngay, không trả số sai. Weights không đưa lên Git.

---

## Hậu kỳ (Module tách riêng)

Hậu kỳ KHÔNG chạy bên trong `OpenStroke.predict()`. Khi cần dựng object từ kết quả raw, gọi riêng ở worker hậu kỳ:
- HVAC polygon: `openstroke.hvac_faces` (`faces_first_hybrid`)
- FIRE / PLUMBING đường ống: `openstroke.line_decoder` (`decode_lines`)
- Hoặc truyền thẳng `r.to_dict()` / `r.save_npz(...)` sang dịch vụ hậu kỳ riêng của hệ thống.

---

## Train

Train trên GPU, cú pháp `key=value` (hoặc `OpenStroke("openstroke46").train(data=..., epochs=20)` / `OpenStroke("openstroke47").train(...)`):

| `model=` | Nghĩa |
|---|---|
| `openstroke46` / `openstroke47` (hoặc `openstroke28`…`openstroke45`) | Kiến trúc theo version, kèm công thức train mặc định, khởi tạo mới |
| `<file>.pt` | Fine-tune: kế thừa kiến trúc & công thức từ checkpoint, cờ dòng lệnh thắng |

```bash
# MỘT kiến trúc openstroke46 duy nhất (tự nhận số lớp & class_names từ dữ liệu huấn luyện!)
# 1. HVAC (nhị phân + object heads)
openstroke train \
    model=openstroke46 \
    data=datasets/e13_v4 \
    epochs=20 \
    name=train_os46_hvac

# 2. FIRE 3 lớp: 0=other / 1=main / 2=branch (hoặc fine-tune trực tiếp model=best_hvac.pt)
openstroke train \
    model=openstroke46 \
    data=datasets/fire_3cls_v1 \
    epochs=25 \
    name=train_os46_fire3
```

---

## Các kiến trúc

| `model=` | Ý chính | Tài liệu |
|---|---|---|
| **`openstroke46`** *(Unified, khuyên dùng)* | **Một kiến trúc vector+ảnh phổ quát duy nhất cho mọi hệ**: `graph_topo_uni` (18D: CC theo style, độ dày trọng số chiều dài, mạch vòng `2-core`, độ sâu tỉa cây, ký hiệu đầu cuối) + `edge_style` 6D GAT + `CADStemV2` giãn nở 3 kênh + **đầu ra động theo dữ liệu (`auto nc`/`class_names`)** + chuyển giao trọng số xuyên hệ | [OPENSTROKE46](docs/OPENSTROKE46.md) · [OPENSTROKE47](docs/OPENSTROKE47.md) |
| `openstroke47` (`openstroke_fire3`) | Preset rút gọn trên cùng thân `openstroke46` thống nhất cho bài toán đa lớp FIRE (`other/main/branch`) | [OPENSTROKE47](docs/OPENSTROKE47.md) |
| `openstroke45` | `openstroke44` + `graph_topo_v2` (CC đồ thị đồng nhất độ dày/màu + thứ bậc độ dày tương đối) + `neg_boxes` v4 | [OPENSTROKE45](docs/OPENSTROKE45.md) |
| `openstroke44` | `openstroke41` + `CADStem` 2048px (độ dày + fill) + `graph_topo` + `GraphAttention` 4 tầng (`rel_attn=4`) | [OPENSTROKE44](docs/OPENSTROKE44.md) |
| `openstroke41` | `openstroke31` + `GroupClsHead` (phân loại cụm 8 lớp + loại FP) + `restyle_mode=robust` | [OPENSTROKE41](docs/OPENSTROKE41.md) |
| `openstroke34` | `openstroke31` + đầu object trên mảnh nét cắt tại góc / giao điểm | [OPENSTROKE34](docs/OPENSTROKE34.md) |
| `openstroke31` | `openstroke30` + loss cạnh cân bằng trên cặp nét đúng | [OPENSTROKE31](docs/OPENSTROKE31.md) |
| `openstroke30` | `openstroke28` + `GraphAttention` 2 tầng + `EdgeHead` + `closure` | [OPENSTROKE30](docs/OPENSTROKE30.md) |
| `openstroke28` | PTv3 U-Net trên shape + `ImageBranch` ResNet18-FPN vẽ từ vector, không đọc layer | [OPENSTROKE28](docs/OPENSTROKE28.md) |

---

## App thử model & Test

```bash
run_app.bat "D:\Models\best_deploy.pt"
python app/hvac_app.py --model models/best_deploy.pt --port 8765
```

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests
```
