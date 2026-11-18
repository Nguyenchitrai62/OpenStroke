# OpenStroke

Phân loại từng nét vector của PDF bản vẽ. Đầu vào là **lệnh vẽ của một trang** (hình học, màu, độ dày, nét đứt, fill); đầu ra là
**kết quả raw: một xác suất cho mỗi shape** là nét cần bóc. "Nét cần bóc" là gì do dữ liệu train quyết định: mỗi bộ dữ liệu train ra một
model riêng, cùng một kiến trúc. Dựng object từ nét là việc của hậu kỳ. Không dùng Ultralytics; package không import PyMuPDF (AGPL).

Kiến trúc mới nhất: **openstroke31**. Phiên bản `0.31.0` (số giữa = kiến trúc mới nhất).

## Kết quả

Model ví dụ E120 (`openstroke30`, ống gió HVAC), val giữ ngoài train: 5 trang GT + 8 trang Viewer. F2 / recall / precision của nét cần bóc
**theo chiều dài nét**, ngưỡng chọn trên val (0,70), TTA ×8.

| Tập | R | P | F2 |
|---|---|---|---|
| **13 trang val, openstroke30** | 0,907 | 0,703 | **0,857** |
| 13 trang val, openstroke28 | 0,886 | 0,693 | 0,839 |
| 5 trang GT, openstroke30 | 0,959–0,975 | 0,689–0,902 | 0,895–0,958 |
| 8 trang Viewer, openstroke30 | 0,70–0,97 | 0,25–0,86 | 0,61–0,86 |

Tham khảo, qua hậu kỳ thử (logic HVAC + khối khép kín): F2 object 5 trang GT 0,827, 8 trang Viewer 0,544.

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

- Repo GitLab công ty cần đăng nhập: `git+https://<user>:<token>@git.anybim.vn/CxDP/worker/modules/openanything/hicasstroke.git`.
- Python ≥ 3.10. GPU RTX 50xx: đổi `cu126` thành `cu130`. Chạy lại đúng lệnh để cập nhật.
- Tuỳ chọn: `openstroke[app]` (app thử model), `openstroke[dev]` (pytest), `openstroke[baseline]` (scikit-learn cho hậu kỳ thử).

## Suy luận

```python
from openstroke.infer import Predictor, load_payload

pred = Predictor("models/hvac_e120.pt")             # nạp một lần, dùng cho mọi trang; GPU nếu có
payload = load_payload("page-0001.v0.json.gz")      # hoặc dict payload đã có trong bộ nhớ
out = pred.predict_payload(payload, mode="hvac", tta=1)
prob = out["prob"]          # np.float32 (số shape,): xác suất shape i là nét cần bóc, đúng thứ tự payload["shapes"]
```

- `out["prob"]` là **raw**: chưa ngưỡng, chưa hậu kỳ. `pred.threshold` là ngưỡng tối ưu F2 nét trên val của model, để tham khảo.
- `out["edges"]` (model `openstroke30`): các cặp shape nối nhau và xác suất chúng thuộc cùng một object.
- `out["kept"]`: số shape có hình học được đưa vào model; shape không có hình học nhận 0.
- `Predictor(checkpoint, device="auto", threads=None, workers=1)`. Trang chia cửa sổ 65.536 shape, bước trượt một nửa.
- `tta=1` nhanh nhất; `tta=pred.tta` (×8) giống lúc val, chậm khoảng 8 lần.

### Đầu vào: payload một trang

JSON định dạng `split_countfire`. Model **chỉ đọc `payload["shapes"]`**:

```python
payload = {"shapes": [
    {"items": [["l", [x0, y0], [x1, y1]], ["c", p0, p1, p2, p3]],     # l đoạn thẳng, c bezier bậc 3, re chữ nhật, qu tứ giác
     "type": "s", "color": [0, 0, 0], "fill": None, "width": 0.35,
     "dashes": "[] 0", "closePath": False, "seqno": 1532, "layer": "M-DUCT"},   # layer không được đọc (no_layers)
]}
```

- Toạ độ: pt, hệ display top-left của trang (đã xoay theo `/Rotate`).
- Một shape = một đơn vị nhãn; không tách hay gộp shape trước khi đưa vào.
- Payload phải sinh bằng cùng cách trích với dữ liệu train.

### Checkpoint

Một model = một file `best.pt`; file tự mang kiến trúc (`config["arch"]`: `openstroke28` / `openstroke29` / `openstroke30` / `openstroke31`), trọng số EMA, bộ
đặc trưng, ngưỡng và TTA lúc val. Code khác bộ đặc trưng thì `predict_payload` báo lỗi, không ra số sai. Weights không đưa lên Git.

## App thử model

```bash
run_app.bat "D:\Models\hvac_e120.pt"
python app/hvac_app.py --model models/hvac_e120.pt --port 8765
```

Thả PDF, xem nét model chọn theo ngưỡng (raw hiện ngay, hậu kỳ thử chạy ngầm). Trích PDF và logic HVAC thử dùng Python của AI_Takeoff
(`tools/extract_payload.py`, `tools/run_hvac_logic.py`, biến `OPENSTROKE_EXTRACTOR_PYTHON`).

## Train

Train trên GPU. Cú pháp `key=value`:

| `model=` | Nghĩa |
|---|---|
| `openstroke31` (hoặc `openstroke28`, `openstroke29`, `openstroke30`) | Kiến trúc theo version, kèm công thức train của nó, khởi tạo mới |
| `<file>.pt` | Train tiếp: kiến trúc và công thức đọc từ checkpoint, cờ trên lệnh thắng |

```bash
openstroke train \
    model=hvac_e114.pt \
    data=datasets/e10 \
    closure=0.5 \
    rel_attn=2 \
    edge_head=true \
    batch=8 \
    lr0=0.001 \
    warmup_epochs=1 \
    epochs=10 \
    name=e120_graph_attn
```

Mỗi fold ghi `runs/<name>/fold<k>/`: `best.pt`, `last.pt`, `result.json`, `val_scores.npz`, `results.csv`, `results.png`.
Dataset: `<stem>.npz` mỗi trang + `manifest.json` + `folds.json` (fold 0 = val, −1 = chỉ train).

## Các kiến trúc

| `model=` | Ý chính | F2 nét (13 trang val) | Thẻ |
|---|---|---|---|
| `openstroke31` | openstroke30, đầu "cùng object" học cân bằng trên cặp nét đúng (tách object cùng lớp nằm sát nhau) | chưa train | [OPENSTROKE31](docs/OPENSTROKE31.md) |
| **`openstroke30`** | openstroke28 + attention theo đồ thị nối đầu mút / khe nét đứt / chữ T + đầu "cùng object" + loss bắt đủ object | **0,857** | [OPENSTROKE30](docs/OPENSTROKE30.md) |
| `openstroke29` | openstroke28 + đầu lớp từng nét (Duct, Elbow, Elbow vuông, Reducer, Chân rẽ, Tee) | 0,842 | [OPENSTROKE29](docs/OPENSTROKE29.md) |
| `openstroke28` | U-Net transformer trên shape (kiểu Point Transformer V3, cửa sổ 65.536) + nhánh ảnh vẽ lại từ vector (ResNet18 + FPN), không đọc layer | 0,839 | [OPENSTROKE28](docs/OPENSTROKE28.md) |

## Test

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests
```

CPU, dữ liệu tổng hợp, không cần GPU hay AI_Takeoff.
