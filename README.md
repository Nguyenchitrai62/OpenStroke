# OpenStroke

Phân loại **từng nét vector** của một trang PDF bản vẽ CAD. Đầu vào là các lệnh vẽ vector của trang (hình học, màu, độ dày,
nét đứt, fill); đầu ra là **xác suất của từng shape** (nhị phân, hoặc `K` lớp tùy dữ liệu train), kèm xác suất hai nét thuộc
cùng một đối tượng nếu checkpoint có đầu ra đó. Train, predict và đọc kết quả theo cùng kiểu với OpenOBB / YOLO.

- Model chỉ trả kết quả raw; dựng đối tượng từ các nét (polygon, tuyến ống...) là bước hậu kỳ riêng, không chạy trong `predict`.
- Ảnh cho nhánh ảnh do model tự vẽ lại từ vector: không cần file PDF hay ảnh trang khi suy luận.
- Không phụ thuộc Ultralytics, package không import PyMuPDF (AGPL); chỉ dùng thành phần MIT / BSD / Apache-2.0.
- Kiến trúc mới nhất: **`openstroke46`** (PTv3 U-Net trên token nét + nhánh ảnh CNN + đồ thị nối giữa các nét; số lớp đầu ra
  tự lấy theo dữ liệu train).

## Kết quả

Tập bản vẽ HVAC không có thông tin layer, 13 trang val giữ ngoài train. F2 theo chiều dài nét (ưu tiên recall), TTA ×8.

| Model | F2 nét | Tài liệu |
|---|---:|---|
| `openstroke45` | **0,948** | [OPENSTROKE45](docs/OPENSTROKE45.md) |
| `openstroke44` | 0,945 | [OPENSTROKE44](docs/OPENSTROKE44.md) |
| `openstroke41` | 0,942 | [OPENSTROKE41](docs/OPENSTROKE41.md) |
| `openstroke46` | chưa có số | [OPENSTROKE46](docs/OPENSTROKE46.md) |

## Cài đặt

**Máy có GPU NVIDIA:**

```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -U "openstroke[torch] @ git+https://github.com/Nguyenchitrai62/OpenStroke.git"
```

**Máy chỉ có CPU:**

```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -U "openstroke[torch] @ git+https://github.com/Nguyenchitrai62/OpenStroke.git"
```

**Chỉ tiền xử lý / hậu kỳ, không cần PyTorch** (ví dụ worker gọi model qua HTTP):

```bash
pip install -U "openstroke @ git+https://github.com/Nguyenchitrai62/OpenStroke.git"
```

- Python ≥ 3.10. GPU RTX 50xx: đổi `cu126` thành `cu130`. Chạy lại đúng lệnh để cập nhật.
- Model tự chạy trên GPU nếu có. GPU mà bản PyTorch đang cài không hỗ trợ thì tự chuyển sang CPU kèm cảnh báo.
- Tùy chọn: `openstroke[dev]` (pytest), `openstroke[app]` (app thử model).

## Đầu vào

Một trang = một dict JSON (đọc được `.json`, `.json.gz` hoặc dict trong bộ nhớ). Model chỉ đọc `payload["shapes"]`:

```python
payload = {
    "shapes": [
        {"items": [["l", [x0, y0], [x1, y1]], ["c", p0, p1, p2, p3]],   # l đoạn thẳng, c bezier bậc 3, re chữ nhật, qu tứ giác
         "type": "s", "color": [0, 0, 0], "fill": None, "width": 0.35,
         "dashes": "[] 0", "closePath": False, "seqno": 1532},
    ],
    "document_metadata": {"page_width": 2384.0, "page_height": 1684.0},
}
```

- Tọa độ tính bằng pt, gốc ở góc trên bên trái của trang đã xoay theo `/Rotate`.
- Một shape là một đơn vị nhãn: không tách hay gộp shape trước khi đưa vào.
- Tên layer không được đọc (các model từ `openstroke28` trở đi train không dùng layer).

## Train

Train trên GPU, cú pháp `key=value` giống YOLO / OpenOBB:

```bash
openstroke train data=datasets/my_data model=openstroke46 epochs=20 name=run1     # -> runs/run1/fold0/best.pt
```

- `data=`: thư mục dataset gồm mỗi trang một file `.npz` (shape + nhãn), `folds.json` (chia train / val theo PDF) và
  `manifest.json` (tên lớp). Số lớp đầu ra lấy từ dataset.
- `model=<tên kiến trúc>` (`openstroke28` … `openstroke47`): khởi tạo mới theo công thức mặc định của kiến trúc đó.
- `model=<file.pt>`: fine-tune; kiến trúc và công thức train lấy từ checkpoint, cờ trên dòng lệnh được ưu tiên.
- Khung train kiểu YOLO: EMA, warmup + cosine LR, AMP, resume, patience, `results.csv` / `results.png`. `best.pt` chọn theo F2 trên val.
- Cuối train tự tạo `best_deploy.pt` / `last_deploy.pt` (chỉ trọng số EMA, fp16, ~35 MB) và in tốc độ deploy với `compile`
  tắt / bật ở cuối log (xem [Tốc độ và `compile`](#tốc-độ-và-compile)).

## Suy luận

```python
from openstroke import OpenStroke

model = OpenStroke("best_deploy.pt")      # nạp một lần: tự gộp Conv + BN, AMP và channels_last trên GPU
model.warmup()                            # tùy chọn, làm ấm một lần khi service khởi động
for r in model.predict(payload):          # dict, file .json / .json.gz, thư mục, glob hoặc list
    r.prob          # (N,) xác suất từng shape theo đúng thứ tự payload["shapes"]
    r.cls_prob      # (N, K) xác suất từng lớp, hoặc None với model nhị phân
    r.class_names   # tên lớp lưu trong checkpoint
    r.valid         # (N,) shape có hình học được chấm (shape rỗng = False)
    r.threshold     # ngưỡng tham khảo chọn trên val (không tự cắt vào r.prob)
    r.edges         # (cặp nét (E, 2), xác suất (E,)) cùng đối tượng, hoặc None
    r.speed         # {"preprocess": ms, "inference": ms, "postprocess": ms}
    r.to_dict(compact=True)                # dict gọn để trả qua HTTP; Results.from_dict(...) để đọc lại
```

```bash
openstroke predict model=best_deploy.pt source=pages/ save=true      # lưu .npz raw mỗi trang
openstroke strip model=runs/run1/fold0/best.pt half=true             # file giao đi gọn (fp16, bỏ tham số train)
openstroke profile model=best_deploy.pt source=page.json.gz          # đo thời gian từng bước trên máy deploy
```

- Nhiều trang tốn ít RAM: `model.predict("pages/", stream=True)`.
- Chỉ cần xác suất nét: `include_groups=False, include_edges=False` để bỏ các đầu ra phụ.
- Endpoint HTTP mẫu: `python -m openstroke.endpoint best_deploy.pt`.

## Tốc độ và `compile`

`compile=True` (hoặc biến môi trường `OPENSTROKE_COMPILE=1`; mặc định tắt, như OpenOBB) chạy `torch.compile` cho phần CNN
của nhánh ảnh. Kết quả không đổi. Lần gọi đầu mất 1–3 phút để biên dịch, các lần sau mới nhanh hơn, nên chỉ hợp với
tiến trình chạy lâu như server.

```python
model = OpenStroke("best_deploy.pt", compile=True)
```

| Máy | `compile=True` |
|---|---|
| GPU NVIDIA sm_70 trở lên (V100, T4, RTX 20xx, A100, L4...) | Chạy |
| GPU cũ hơn sm_70 | Bỏ qua kèm cảnh báo (Triton cần sm_70+), chạy bản thường |
| CPU | Bỏ qua kèm cảnh báo, chạy bản thường. Biên dịch trên CPU cần trình biên dịch C++ (Windows: MSVC `cl`, Linux: `g++`); OpenOBB cũng chạy bản thường khi thiếu trình biên dịch |

**Trên GPU:** cuối mỗi lần `openstroke train`, trainer nạp `best_deploy.pt` như khi deploy, chạy các trang val với compile tắt
rồi bật, và in bảng ở những dòng cuối của log (cũng lưu trong `result.json`, khóa `deploy_speed`). Không cần thêm cờ:

```text
Deploy speed: best_deploy.pt, 13 val pages, TTA x1, model stage only (<GPU>)
  page                                 shapes   off ms    on ms  speedup  1st call ms
  ...
  total                                   ...      ...      ...    ...x          ...
  compile=True: ...x on the model stage after a ... s first pass (compilation); max|dp| vs off ...
```

`off ms` / `on ms`: thời gian bước model mỗi trang (trung bình 2 lượt sau khi làm ấm). `1st call ms`: lượt đầu, gồm thời gian
biên dịch. `max|dp|`: chênh lệch xác suất lớn nhất giữa hai chế độ.

**Trên CPU** (compile tắt, TTA ×1, torch 2.14.1+cpu, Intel Core thế hệ 13/14, 10 luồng, Windows 11; trang A1 2384 × 1684 pt):

| Trang | Shape | Mỗi trang | Tiền xử lý | Model | Hậu xử lý raw |
|---|---:|---:|---:|---:|---:|
| Nhỏ | 8.965 | **1,0 s** | 0,12 s | 0,87 s | 0,01 s |
| Lớn | 338.458 | **32,8 s** | 2,3 s | 30,0 s | 0,5 s |

Thời gian CPU gần như nằm hết ở bước model và tăng theo số shape. Đo lại trên máy của bạn (mỗi chế độ chạy một tiến trình
riêng; khi máy có trình biên dịch C++, script đo luôn chế độ compile bật và kiểm tra kết quả hai chế độ khớp nhau):

```bash
python tools/bench_cpu_compile.py --model best_deploy.pt --pages small.json.gz large.json.gz --repeat 3
```

## Các kiến trúc

| `model=` | Kiến trúc | Tài liệu |
|---|---|---|
| **`openstroke46`** | Một kiến trúc cho mọi bộ dữ liệu: đồ thị nối 18 chiều, GAT theo kiểu nét, nhánh ảnh CAD 3 kênh, đầu ra `K` lớp theo dataset | [OPENSTROKE46](docs/OPENSTROKE46.md) |
| `openstroke45` | v44 + đồ thị nối lọc theo cùng độ dày / màu | [OPENSTROKE45](docs/OPENSTROKE45.md) |
| `openstroke44` | v41 + nhánh ảnh CAD 2048 px + đặc trưng đồ thị + attention theo quan hệ | [OPENSTROKE44](docs/OPENSTROKE44.md) |
| `openstroke41` | v31 + đầu phân lớp theo cụm nét | [OPENSTROKE41](docs/OPENSTROKE41.md) |
| `openstroke30` | v28 + attention trên đồ thị nối + đầu "cùng đối tượng" cho cặp nét | [OPENSTROKE30](docs/OPENSTROKE30.md) |
| `openstroke28` | PTv3 U-Net + nhánh ảnh ResNet18-FPN, không dùng layer | [OPENSTROKE28](docs/OPENSTROKE28.md) |

Mỗi thẻ trong `docs/` ghi giả thuyết, kiến trúc, lệnh train và kết quả thật của phiên bản đó.

## Test

Chạy trên CPU với dữ liệu tổng hợp, không cần GPU:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests
```

## Cấu trúc repo

| Thư mục | Nội dung |
|---|---|
| `openstroke/` | Thư viện: `infer.py` (`OpenStroke`), `results.py`, `models/`, `engine/` (train), `strip.py`, `cli.py` |
| `openstroke/hvac_faces.py`, `line_decoder.py` | Hậu kỳ tùy chọn dựng đối tượng từ kết quả raw (không chạy trong `predict`) |
| `tests/` | `pytest -q tests` (chạy CPU) |
| `docs/` | Thẻ kiến trúc từng phiên bản |
| `tools/` | Script dữ liệu, đo tốc độ, phát hành |
| `app/` | App thử model |

## License

Proprietary, xem [LICENSE](LICENSE). Thành phần bên thứ ba (MIT, BSD-3-Clause, Apache-2.0): [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
