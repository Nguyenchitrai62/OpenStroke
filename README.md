# OpenStroke

Phân loại từng nét vector của PDF bản vẽ. Đầu vào là **lệnh vẽ vector của một trang** (hình học, màu, độ dày, nét đứt, fill); đầu ra là
**kết quả raw: xác suất từng shape** (kèm xác suất cạnh cùng object và điểm cụm nếu checkpoint có đầu tương ứng). **Model không phụ thuộc hệ (`mode`)** — "nét cần bóc" là gì do dữ liệu huấn luyện quyết định; hậu kỳ dựng polygon / tuyến ống nằm ở module riêng để có thể đóng gói & host model độc lập (tương tự kiến trúc triển khai của OpenOBB trong AI_Takeoff).

Không dùng Ultralytics; package không import PyMuPDF (AGPL). Kiến trúc mới nhất: **`openstroke46`** — **một kiến trúc vector+ảnh phổ quát duy nhất** (`PTv3 U-Net + CADStemV2 2048px + 4-layer edge_style GAT + 18D graph_topo_uni`) với **đầu ra động theo dữ liệu huấn luyện** (tự nhận nhị phân hoặc `K` lớp bất kỳ + `class_names` từ `manifest.json`, chuyển giao trọng số xuyên hệ HVAC ↔ FIRE ↔ N lớp). Phiên bản **`0.48.0`** (số giữa = bản phát hành kiến trúc mới nhất, như OpenOBB).

---

## Kiến trúc triển khai (tách hẳn Model Raw & Hậu kỳ)

| Thành phần | Vai trò | Triển khai |
|---|---|---|
| **`openstroke`** (`OpenStroke`) | **Load model → Infer vector payload → Trả kết quả raw** (`prob`, `valid`, `edges`, `groups`, `speed`). Không cần truyền `mode`; không import hay chạy hậu kỳ. Ảnh cho nhánh ảnh do model tự vẽ lại từ vector, không cần file PDF | Host thành service độc lập (GPU/CPU), giao tiếp JSON/NPZ |
| **Module hậu kỳ riêng** (`openstroke.hvac_faces`, `openstroke.line_decoder`, hoặc logic sản phẩm của AI_Takeoff) | Nhận kết quả raw (`prob`, `edges`, `groups`) + hình học payload → dựng polygon object HVAC hoặc tuyến đường ống FIRE/PLUMBING | Chạy ở worker / dịch vụ hậu kỳ riêng, nâng cấp độc lập với weights |
| `tools/extract_payload.py` (`pdf_etl`) | **Chỉ để test local**: trích lệnh vẽ từ PDF thành payload | Chạy bằng Python AI_Takeoff (PyMuPDF tách ngoài package) |

---

## Suy luận & Triển khai (Giao diện kiểu OpenOBB)

### Python API (khuyên dùng cho deploy)

```python
from openstroke import OpenStroke

# 1. Nạp model một lần khi khởi động service (tự nhận GPU bf16 + gộp Conv/BN + channels_last, hoặc CPU)
model = OpenStroke("models/best_deploy.pt")
model.warmup()                                  # làm ấm CUDA / thread pool một lần lúc boot

# 2. Suy luận raw — KHÔNG cần truyền mode (nhận dict payload, file .json / .json.gz, thư mục, glob, hoặc list)
results = model.predict(payload)                # hoặc gọi tắt: results = model(payload)

for r in results:
    r.prob          # np.float32 (N,): xác suất RAW từng shape theo đúng thứ tự payload["shapes"] (với model K lớp = 1 - p_other)
    r.cls_prob      # np.float32 (N, K) | None — xác suất từng lớp raw (động theo dữ liệu huấn luyện, vd. [p_other, p_main, p_branch])
    r.class_names   # tuple[str, ...] | None — tên lớp tương ứng từ checkpoint (vd. ("other", "main", "branch"))
    r.valid         # np.bool_   (N,): mặt nạ shape có hình học đã được chấm (shape rỗng = False, prob = 0)
    r.threshold     # float: ngưỡng tham khảo từ val (KHÔNG tự cắt vào r.prob)
    r.edges         # (pairs (E, 2), prob (E,)) | None — xác suất cặp nét nối thuộc cùng object (nếu model có edge_head)
    r.groups        # dict | None — cụm + xác suất 8 lớp của GroupClsHead (nếu model có group_head)
    r.speed         # {"preprocess": ms, "inference": ms, "postprocess": ms}
    raw_resp = r.to_dict(compact=True)          # dict nén thưa (~10-20x nhỏ hơn trên trang A0) trả về qua HTTP / RPC
```

**Phía Caller (service hậu kỳ riêng nhận `raw_resp` từ API model):**

```python
from openstroke import Results
from openstroke.hvac_faces import postprocess_hvac       # hoặc: from openstroke.line_decoder import postprocess_lines

raw = Results.from_dict(raw_resp)                        # bung lại đầy đủ prob / valid / edges / groups
hvac = postprocess_hvac(payload, raw, classifier_path="models/hvac_faces.joblib")
# FIRE / PLUMBING: pipes = postprocess_lines(payload, raw, mode="fire")
```

- **Trả raw siêu gọn khi chỉ cần xác suất nét:** truyền `include_groups=False` (hoặc thêm `include_edges=False`) để bỏ qua bước giải mã cụm/cạnh bổ sung.
- **Xử lý nhiều trang tốn ít RAM:** dùng `stream=True` (`for r in model.predict("pages/", stream=True): ...`) — đọc từng file payload một lần thay vì nạp cả thư mục vào RAM.
- **Tối ưu tốc độ deploy:**
  - Nạp checkpoint tự động **gộp `Conv2d + FrozenBatchNorm2d`** trong `ImageBranch`, bật **`channels_last` + TF32 + `cudnn.benchmark` + AMP bf16/fp16** trên GPU (và `flush_denormal` trên CPU).
  - `render_multi` vẽ chung tọa độ pixel 1 lần cho cả 3 kênh CAD (`ink`, `width`, `has_fill` ở `openstroke44`/`45`), song song hóa tiền xử lý (`tokens + shape_points + endpoint/gap/tee graph`), và loại bỏ các đồng bộ CUDA thừa (`torch.unique` / `grp.max()`) dưới `no_layers=True`.
  - Bật `compile=True` (hoặc `OPENSTROKE_COMPILE=1`) cho server chạy lâu để `torch.compile` nhánh ảnh trên GPU (mặc định tắt, như OpenOBB).
- **Tương thích ngược:** `openstroke.infer.Predictor` vẫn hoạt động đầy đủ (`mode` đã trở thành tham số tùy chọn, mặc định `None`).

### CLI kiểu YOLO / OpenOBB

```bash
# Suy luận raw trên 1 file, thư mục hoặc glob (lưu .npz / .json)
openstroke predict model=best.pt source=pages/ save=true

# Cắt gọn checkpoint giao đi (chỉ giữ trọng số EMA + cấu hình mạng, bỏ đường dẫn/tham số train; half=true giảm ~50% dung lượng)
openstroke strip model=runs/e150/fold0/best.pt half=true

# Đo tốc độ thực tế từng bước (preprocess / inference / raw-head) trên máy deploy
openstroke profile model=best_deploy.pt source=page-0001.v0.json.gz
```

---

## Cài đặt

**Máy có GPU NVIDIA:**

```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -U "openstroke @ git+https://github.com/Nguyenchitrai62/OpenStroke.git"
```

**Máy chỉ có CPU:**

```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -U "openstroke @ git+https://github.com/Nguyenchitrai62/OpenStroke.git"
```

- Python ≥ 3.10. GPU RTX 50xx: đổi `cu126` thành `cu130`. Chạy lại đúng lệnh để cập nhật.
- Tuỳ chọn: `openstroke[app]` (app thử model), `openstroke[dev]` (pytest), `openstroke[baseline]` (`scikit-learn` cho hậu kỳ test / GBDT).
- Làm việc trên mã nguồn:

```bash
git clone https://github.com/Nguyenchitrai62/OpenStroke.git
cd OpenStroke
pip install -e ".[app,dev]"
```

---

## Quy chuẩn đầu vào & đầu ra

### Đầu vào: payload một trang

JSON định dạng `split_countfire` (đọc được cả `.json` lẫn `.json.gz`, hoặc `dict` trong bộ nhớ). Model **chỉ đọc `payload["shapes"]`**:

```python
payload = {
    "shapes": [
        {"items": [["l", [x0, y0], [x1, y1]], ["c", p0, p1, p2, p3]],   # l đoạn thẳng, c bezier bậc 3, re chữ nhật, qu tứ giác
         "type": "s", "color": [0, 0, 0], "fill": None, "width": 0.35,
         "dashes": "[] 0", "closePath": False, "even_odd": False,
         "stroke_opacity": 1.0, "fill_opacity": 1.0, "seqno": 1532, "layer": "M-DUCT"},
    ],
}
```

- **Toạ độ:** pt, hệ display top-left (đã xoay theo `/Rotate`), đúng như `pdf_etl.run_pipeline`.
- **Một shape = một đơn vị nhãn.** Không tách hay gộp shape trước khi đưa vào.
- **`layer` không được đọc** trên các model không layer (`no_layers=True`, từ `openstroke28` trở đi).
- **Kiểm tra đầu vào:** `OpenStroke.predict` kiểm tra hợp lệ `payload["shapes"]` ngay từ đầu và trả kèm mặt nạ `r.valid` đánh dấu các shape thực sự có hình học được chấm.

### Checkpoint

Một model = một file `.pt` (dùng thẳng `best.pt` từ `openstroke train`, hoặc file gọn `best_deploy.pt` từ `openstroke strip`). Checkpoint tự mang đầy đủ `config`, trọng số EMA, `feature_names`, `window`, `threshold`, `tta`:

```python
from openstroke import OpenStroke
OpenStroke("models/best_deploy.pt").describe()
```

---

## Kết quả gần nhất (HVAC không layer, 13 trang val giữ ngoài `GT5 + VW8-v4`)

Số liệu thật đo trên tập val giữ ngoài `score_pack_v4_val` (5 trang GT + 8 trang Viewer đã kiểm toán trực quan), TTA ×8:

| Model / Run | Kiến trúc | F2 nét (`all` / `GT5` / `VW8`) | Thuần model `6cls` (`GT5` / `VW8`) | Hậu kỳ tham khảo (`GT5` / `VW8`) | Thẻ |
|---|---|---:|---:|---:|---|
| **`openstroke46`** (Unified, `E151` HVAC / `E152` FIRE 3 lớp / N lớp) | **Kiến trúc phổ quát duy nhất**: `graph_topo_uni` (18D) + `edge_style` 6D GAT + `CADStemV2` giãn nở 3ch + **đầu ra động `auto nc`/`class_names`** + `best.pt` mới | *chờ log GPU* | *chờ log GPU* | *chờ log GPU* | [OPENSTROKE46](docs/OPENSTROKE46.md) · [OPENSTROKE47](docs/OPENSTROKE47.md) |
| `openstroke45` (`E150`) | OS44 + `graph_topo_v2` (CC cùng độ dày/màu + thứ bậc độ dày) + `neg_boxes` v4 | **0,948** (**0,946** / **0,950**) | 0,764 / 0,777 | `POLY` **0,911** / 0,776 (v7, OOD `last.pt` tốt nhất) | [OPENSTROKE45](docs/OPENSTROKE45.md) |
| `openstroke44` (`E149`) | OS41 + `CADStem` (`2048 px`) + `graph_topo` + `rel_attn=4` trên `e13_v4` | 0,945 (0,943 / 0,946) | **0,768 / 0,780** | `POLY` **0,912** / 0,777 (v7) | [OPENSTROKE44](docs/OPENSTROKE44.md) |
| `openstroke41` (`E145`) | OS31 + `GroupClsHead` (8 lớp + FP veto) + `restyle_mode=robust` | 0,942 (0,933 / 0,947) | 0,759 / 0,762 | `POLY` 0,905 / **0,788** (v7) | [OPENSTROKE41](docs/OPENSTROKE41.md) |
| `openstroke30` (`E120`) | OS28 + `GraphAttention` 2 tầng + `EdgeHead` + `closure` (nhãn `e10` cũ) | 0,857 | — | 0,827 / 0,544 (hậu kỳ cũ) | [OPENSTROKE30](docs/OPENSTROKE30.md) |
| `openstroke28` (`E114`) | PTv3 U-Net + `ImageBranch` ResNet18-FPN, không layer (nhãn `e8` cũ) | 0,839 | — | — | [OPENSTROKE28](docs/OPENSTROKE28.md) |

*(Ghi chú: không so chéo số liệu trên các bộ nhãn khác nhau `e8`/`e10` vs `e13_v4`; `E150` đạt kỷ lục nét `0,948` và `E150/fold0/last.pt` thắng toàn diện OOD mù, đồng thời để lộ 4 nút thắt đã xử lý trực tiếp trong `openstroke46`.)*

---

## Train

Train trên GPU (Colab), cú pháp `key=value` giống YOLO / OpenOBB (cũng có thể gọi `OpenStroke("openstroke46").train(data=..., epochs=20)` hoặc `OpenStroke("openstroke47").train(...)`):

```bash
# MỘT kiến trúc openstroke46 cho mọi hệ (tự nhận số lớp & class_names từ dữ liệu huấn luyện!)
# 1. HVAC (nhị phân + object heads)
openstroke train \
    data=/content/os44/e13_v4 \
    model=openstroke46 \
    epochs=20 \
    name=e151_os46_unified_hvac

# 2. FIRE 3 lớp: 0=other / 1=main / 2=branch (từ đầu hoặc transfer thẳng từ checkpoint HVAC!)
openstroke train \
    data=/content/fire3/fire_3cls_v1 \
    model=openstroke46 \
    epochs=25 \
    name=e152_os46_fire3
```

- `model=<tên kiến trúc>` (`openstroke28` … `openstroke47`): khởi tạo mới theo công thức mặc định của thẻ nghiên cứu.
- `model=<file.pt>`: fine-tune, tự kế thừa kiến trúc & công thức từ checkpoint, cờ truyền trên dòng lệnh thắng.
- Lệnh Colab chi tiết: [docs/OPENSTROKE46.md](docs/OPENSTROKE46.md) (HVAC) & [docs/OPENSTROKE47.md](docs/OPENSTROKE47.md) (FIRE 3 lớp).

### App thử model local (gồm hậu kỳ kiểm thử)

```bash
run_app.bat "D:\Models\best.pt"
python app/hvac_app.py --model models/best.pt --port 8765 [--no-browser]
```

App cần Python của AI_Takeoff (`OPENSTROKE_EXTRACTOR_PYTHON`) chỉ để trích PDF và chạy so sánh hậu kỳ thử; package suy luận `openstroke` không phụ thuộc vào app hay AI_Takeoff.

---

## Test

Chạy CPU, dữ liệu tổng hợp, không cần GPU hay AI_Takeoff:

```bash
OMP_NUM_THREADS=1 py -3.12 -m pytest -q tests
```

---

## Cấu trúc repo & Phát hành

| Thư mục | Nội dung |
|---|---|
| `openstroke/` | Thư viện cốt lõi: `infer.py` (`OpenStroke`, `Predictor`), `results.py` (`Results`, `Strokes`), `strip.py`, `features.py`, `graph.py`, `models/`, `engine/`, `cli.py` |
| `openstroke/hvac_faces.py`, `line_decoder.py` | Module hậu kỳ độc lập (không import khi chạy `OpenStroke.predict`) |
| `openstroke/obb/` | Hướng cũ (detector OpenOBB nhận vector, chỉ trên GitHub, không đẩy sang bản sản phẩm GitLab) |
| `app/`, `tools/`, `docs/`, `tests/` | App thử nghiệm, công cụ trích/đồng bộ, thẻ kiến trúc, unit tests |
| `research/` | Sổ cái, giả thuyết, probe đo (**chỉ ở máy nghiên cứu**, `.gitignore`) |

Phát hành đồng bộ lên GitHub và GitLab sản phẩm:

```bash
python tools/push_all.py              # push GitHub + tag, sync bản sản phẩm -> pytest -> build wheel -> push GitLab + tag
python tools/push_all.py --dry-run    # kiểm tra diff và chống rò rỉ trước khi đẩy
```

## License

Proprietary, xem [LICENSE](LICENSE). Thành phần bên thứ ba (MIT, BSD-3-Clause, Apache-2.0): [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
