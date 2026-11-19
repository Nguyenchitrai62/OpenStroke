# OpenStroke

Phân loại từng nét vector của PDF bản vẽ. Đầu vào là **lệnh vẽ của một trang** (hình học, màu, độ dày, nét đứt, fill); đầu ra là
**kết quả raw: một xác suất cho mỗi shape** là nét cần bóc. "Nét cần bóc" là gì do dữ liệu train quyết định: mỗi bộ dữ liệu train ra một
model riêng, cùng một kiến trúc. Dựng object từ nét là việc của hậu kỳ, nằm ngoài OpenStroke. Không dùng Ultralytics; package không import
PyMuPDF (AGPL). Kiến trúc mới nhất: **openstroke34**. Phiên bản `0.34.0` (số giữa = kiến trúc mới nhất, như OpenOBB).

> **Trạng thái: nghiên cứu.** Chưa cắm vào hệ thống thật. Số liệu dưới đây lấy từ log train, ghi rõ tập trang.

## Phạm vi

| Phần | Vai trò |
|---|---|
| **`openstroke`** (package) | Lệnh vẽ → xác suất từng shape (raw). Ảnh cho nhánh ảnh do model tự vẽ lại từ vector, không cần file PDF |
| `tools/extract_payload.py` (`pdf_etl`) | **Chỉ để test**: trích lệnh vẽ từ PDF thành payload cho app và thí nghiệm |
| `tools/run_hvac_logic.py`, hậu kỳ trong `openstroke/` | **Chỉ để test / so sánh**: dựng object từ raw để chấm F2 object. Không phải hậu kỳ sản phẩm |

## Kết quả

Ví dụ một model đã train: **E120** (`openstroke30`, bộ dữ liệu ống gió HVAC), val giữ ngoài train (5 trang GT + 8 trang Viewer).
F2 / recall / precision của nhãn `true` **theo chiều dài nét**, ngưỡng chọn trên val (0,70), TTA ×8.

| Tập | R | P | F2 |
|---|---|---|---|
| **13 trang val, openstroke30 (E120)** | 0,907 | 0,703 | **0,857** |
| 13 trang val, openstroke28 (E114) | 0,886 | 0,693 | 0,839 |
| 5 trang GT (doc180 / 187 / 189 / 200 / 201), E120 | 0,959–0,975 | 0,689–0,902 | 0,895–0,958 |
| 8 trang Viewer (kiểu vẽ khác, nhãn `in_progress`), E120 | 0,70–0,97 | 0,25–0,86 | 0,61–0,86 |
| Gán `true` mọi nét (mốc) | 1,000 | – | 0,168 |

- 17,2M tham số. Train 10 epoch từ E114, 0,81 h, 82,4 GB VRAM trên Colab. Model không đọc layer PDF.
- Tham khảo, qua hậu kỳ test (logic HVAC + khối khép kín, không phải hậu kỳ sản phẩm): F2 object 5 trang GT 0,827, 8 trang Viewer 0,544.
- Điểm yếu: vài trang Viewer chỉ giữ 70–75 % nét đúng; ô chú thích hình chữ nhật đôi khi bị gọi là nét cần bóc với xác suất cao.
- Chi tiết: [docs/OPENSTROKE30.md](docs/OPENSTROKE30.md).

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
- Tuỳ chọn: `openstroke[app]` (app thử model), `openstroke[dev]` (pytest), `openstroke[baseline]` (`baseline-gbdt`).
- Làm việc trên mã nguồn (app, `tools/`, `research/` không nằm trong gói pip):

```bash
git clone https://github.com/Nguyenchitrai62/OpenStroke.git
cd OpenStroke
pip install -e ".[app,dev]"
```

Suy luận raw chỉ cần package này. Công cụ test (trích PDF, logic HVAC) chạy bằng Python của AI_Takeoff, xem [Công cụ test](#công-cụ-test).

## Quy chuẩn đầu vào và đầu ra

### Đầu vào: payload một trang

Payload là JSON định dạng `split_countfire` (cũng là input mà SDK HVAC / FIRE của AI_Takeoff nhận). Model **chỉ đọc `payload["shapes"]`**.

```python
payload = {
    "shapes": [
        {"items": [["l", [x0, y0], [x1, y1]], ["c", p0, p1, p2, p3]],   # lệnh vẽ của shape
         "type": "s", "color": [0, 0, 0], "fill": None, "width": 0.35,
         "dashes": "[] 0", "closePath": False, "even_odd": False,
         "stroke_opacity": 1.0, "fill_opacity": 1.0, "seqno": 1532, "layer": "M-DUCT"},
        ...
    ],
}
```

| Trường | Ý nghĩa |
|---|---|
| `items` | Danh sách lệnh vẽ: `l` (đoạn thẳng, 2 điểm), `c` (bezier bậc 3, 4 điểm), `re` (chữ nhật), `qu` (tứ giác). Lệnh khác bị bỏ qua |
| `type` | `s` nét, `f` tô, `fs` tô + nét |
| `color`, `fill` | RGB trong [0, 1]; `None` nếu không có |
| `width` | Độ dày nét, pt |
| `dashes`, `closePath`, `even_odd`, `stroke_opacity`, `fill_opacity`, `seqno` | Thuộc tính vẽ như PyMuPDF trả về; `seqno` là thứ tự vẽ |
| `layer` | Model train với `no_layers=true` **không đọc**; có hay không đều được |

- **Toạ độ:** pt, hệ display top-left của trang (đã xoay theo `/Rotate`), đúng như `pdf_etl.run_pipeline` trả ra.
- **Một shape = một đơn vị nhãn.** Không tách hay gộp shape trước khi đưa vào: nét đứt CAD đã bung thành nhiều shape ngắn, mỗi shape một xác suất.
- Payload phải sinh bằng cùng cách trích với dữ liệu train. Đổi cách trích (ví dụ tách nét đứt khác đi) thì phải chấm lại model.

### Đầu ra

```python
out = pred.predict_payload(payload, mode="hvac")   # mode = hệ model đã train (args["modes"])
out["prob"]      # np.float32, shape (len(payload["shapes"]),): xác suất shape i là nét cần bóc, đúng thứ tự payload
out["kept"]      # số shape có hình học, được đưa vào model; shape không có hình học nhận 0
out["seconds"]   # {"features": s, "model": s}
out["info"]      # segments, attrs, length... của trang (dùng cho hậu kỳ test)
```

- `prob` là **raw**: chưa ngưỡng, chưa hậu kỳ. Hậu kỳ tự chọn ngưỡng theo mục tiêu của nó.
- `pred.threshold` là ngưỡng tối ưu F2 nét trên val của model đó, chỉ để tham khảo.

### Checkpoint

Một model = một file `best.pt` (ghi bởi `openstroke train` ở `runs/<name>/fold<k>/best.pt`). File tự mang mọi thứ suy luận cần:

| Khoá | Nội dung |
|---|---|
| `model`, `config` | Trọng số (bản EMA) và cấu hình kiến trúc, gồm `no_layers`, `mode_input` |
| `args` | Tham số train (cửa sổ, `modes`...) |
| `feature_names` | Bộ đặc trưng đã train. Code khác bộ đặc trưng thì `predict_payload` báo lỗi, không ra số sai |
| `threshold`, `tta`, `val_f2`, `epoch` | Ngưỡng chọn trên val, số TTA lúc val, F2 val, epoch tốt nhất |

```python
from openstroke.infer import Predictor
Predictor("models/hvac_e120.pt").describe()   # arch, device, window, threshold, tta_used_in_val, val_f2, epoch
```

Quy ước:
- Weights **không đưa lên Git**. Đặt tên theo bộ dữ liệu và run, ví dụ `models/hvac_e120.pt`.
- Mỗi checkpoint đi cùng một phiên bản code `openstroke` (commit). Đổi code phần đặc trưng thì train lại.
- Truyền `mode` đúng hệ model đã train (`args["modes"]`).

## Suy luận

### Python (raw)

```python
from openstroke.infer import Predictor, load_payload

pred = Predictor("models/hvac_e120.pt")   # nạp một lần, dùng cho mọi trang
payload = load_payload("page-0001.v0.json.gz")   # hoặc dict payload đã có trong bộ nhớ
out = pred.predict_payload(payload, mode="hvac", tta=1)
prob = out["prob"]
```

- `Predictor(checkpoint, device="auto", threads=None, workers=1)`: `device` `"auto"` | `"cuda"` | `"cpu"`. GPU gom 4 cửa sổ một batch, chạy bf16;
  hết VRAM thì tự lùi về 1 cửa sổ. CPU chạy fp32; `threads` đặt số luồng torch, `workers=2` chạy hai cửa sổ song song.
- Trang được chia thành cửa sổ 65.536 shape, bước trượt một nửa; shape nằm ở hai cửa sổ lấy trung bình.
- `tta=1` nhanh nhất. `tta=pred.tta` (×8: lật / quay 90°) giống lúc val, chậm khoảng 8 lần. Số kết quả ở trên đo với ×8; chênh lệch với `tta=1` chưa đo.
- CPU, E120, TTA ×1: trang 37 nghìn shape khoảng 8 s, 390 nghìn shape khoảng 2,5 phút (kể cả dựng đồ thị nối lần đầu).

### App thử model

```bash
run_app.bat                                :: model mặc định models\openstroke28.pt
run_app.bat "D:\Models\hvac_e120.pt"      :: hoặc kéo thả file .pt vào run_app.bat
```

```bash
python app/hvac_app.py --model models/hvac_e120.pt --port 8765 [--no-browser]
```

Trình duyệt mở `http://127.0.0.1:8765`. Thả PDF, mọi trang chạy nền theo thứ tự (trang đang mở được ưu tiên), hiện thời gian từng bước.
Kéo ngưỡng đổi vùng chọn ngay, không chạy lại model; có histogram xác suất, chế độ xem raw / heat / logic test / song song.
App cần Python của AI_Takeoff để trích PDF và chạy logic test; logic test hiện viết cho ống gió HVAC. `app/streamlit_app.py` là app cũ.

### Chấm cả dataset

```bash
openstroke scores model=runs/e120_graph_attn/fold0/best.pt data=datasets/e10
```

Chạy `best.pt` trên mọi trang (train + val) với đúng cửa sổ, bước trượt và TTA của lượt val cuối run, ghi `fold0/all_scores.npz`
(định dạng `val_scores.npz`, kèm split từng trang; `openstroke30` ghi thêm xác suất "cùng object" của từng cặp nét nối nhau).
Dùng để làm hậu kỳ; kết luận chỉ lấy trên trang val.

### Công cụ test

Hai công cụ cần PyMuPDF (AGPL) nên chạy bằng Python của AI_Takeoff qua tiến trình con, không import vào package:

| Công cụ | Lệnh |
|---|---|
| PDF → payload | `<python AI_Takeoff> tools/extract_payload.py page <file.pdf> <trang> <out.json.gz>` (còn `info`, `preview`, `thumbs`) |
| Raw → object (logic HVAC dùng chung của AI_Takeoff) | `<python AI_Takeoff> tools/run_hvac_logic.py <payload> <choice.json> <file.pdf> <trang> <out.json>` |

`choice.json` của logic: `{"selected": [chỉ số shape có p ≥ ngưỡng], "how": "style"}` (`style` = mỗi cặp màu + độ dày một layer, `one` = một
layer, `real` = layer PDF). App đọc Python đó từ biến `OPENSTROKE_EXTRACTOR_PYTHON`
(mặc định `F:\Source_code\AI_Takeoff\BE\.venv\Scripts\python.exe`).

## Dữ liệu

- **Nguồn nhãn:** chỉ GT người sửa của AI_Takeoff (group `primary` / `reference`) và bộ nhãn user cung cấp. Không dùng output pipeline / model
  làm nhãn. DB, payload, cache của AI_Takeoff **chỉ đọc**. Dữ liệu dẫn xuất nằm trong `datasets/` (gitignored), không commit.
- **Quy tắc nhãn** (chi tiết [AGENTS.md](AGENTS.md) mục 2, code `openstroke/relabel.py` `object_strokes`): nhãn là các nét tạo nên object GT.
  Shape `true` khi ≥ 50 % chiều dài cách cạnh object GT ≤ 2 pt và lệch hướng ≤ 20°; object dạng polygon còn yêu cầu nét nằm trong polygon,
  object dạng đường đo dọc đường. Bezier lấy mẫu dọc cung.
- **Chia tập:** theo PDF, cả dự án ở cùng một phía; bản trùng, trang cắt, augmentation không rò sang phía khác. Đóng băng manifest và seed
  trước khi train. Ngưỡng, epoch, hyperparameter chọn trên val.

Bố cục một dataset: `<stem>.npz` mỗi trang (nhãn, đoạn thẳng, thuộc tính shape) + `manifest.json` + `folds.json` (fold 0 = val, −1 = chỉ train).

Ví dụ dựng dataset `e8` của E114 (19 trang GT + 171 trang Viewer train; val 5 GT + 8 Viewer):

```bash
py -3.12 tools/make_stroke_dataset.py datasets/obb_hvac10b datasets/viewer_hvac19/exports_20261113/labels.json datasets/e8
```

Script này hiện cố định hệ `hvac` và split `obb_hvac10b`; bộ dữ liệu khác cần bản nhận tham số hệ và split.
Lệnh dựng dataset cũ: `openstroke build-e0`, `relabel-v3`, `build-unlabeled`, `merge-data`, `pack` (`py -3.12 -m openstroke <lệnh> -h`).

## Train

Train trên GPU (Colab). Cú pháp `key=value` như YOLO / OpenOBB (`--key value` cũng được). Bắt buộc có `model=`:

| `model=` | Nghĩa |
|---|---|
| `openstroke34` (hoặc `openstroke28`, `openstroke29`, `openstroke30`, `openstroke31`) | Kiến trúc theo version nghiên cứu, kèm công thức train của thẻ đó, khởi tạo mới |
| `<file>.pt` | Train tiếp / fine-tune: kiến trúc và công thức train đọc từ checkpoint, cờ trên lệnh thắng |
| `<thư mục run>` | Mỗi fold fine-tune từ `fold<k>/best.pt` của chính nó |

Ví dụ lệnh của E120 (`openstroke30`), fine-tune từ E114; bộ dữ liệu khác chỉ đổi `data=`, `name=`:

```bash
openstroke train \
    model=e114.pt \
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

Lệnh Colab đầy đủ (cài từ GitHub, giải nén dữ liệu, lưu run lên Drive): [docs/OPENSTROKE30.md](docs/OPENSTROKE30.md) §10.

Mỗi fold ghi vào `runs/<name>/fold<k>/`: `best.pt`, `last.pt` (để `resume`), `result.json`, `val_scores.npz`, `results.csv`, `results.png`;
run ghi `runs/<name>/train.log`. Dòng `start` / `model` đầu log in tên kiến trúc (`openstroke30` ...) và công thức đọc từ checkpoint.
Log mỗi epoch in F2 nét trên val. Khung train: AdamW, warmup + cosine (`lr0`, `lrf`), EMA, AMP bf16, `patience`, `resume=<run>`, TTA ×8 ở val cuối.

| Cờ | Tác dụng |
|---|---|
| `modes=hvac` | Hệ của trang dùng để train và val (`fire`, `hvac`, `plumbing`, danh sách cách nhau dấu phẩy) |
| `no_layers=true` | Model không đọc layer PDF: nét vẽ sai layer có cùng đầu vào như khi đúng layer |
| `image_branch=true`, `img_fpn=true`, `img_lum=true` | Nhánh ảnh vẽ lại từ vector, ResNet18 + FPN tới stride 4, mỗi nét theo độ đậm màu |
| `rel_attn=2` | (openstroke30) 2 khối attention theo đồ thị nối đầu mút / khe nét đứt / chữ T, khởi tạo 0 |
| `edge_head=true`, `edge_loss=0.5` | (openstroke30) Đầu "cùng object" cho mỗi cặp nét nối nhau |
| `closure=0.5` | (openstroke30) Loss bắt đủ object: soft-min logit các nét của mỗi object GT về 1 |
| `obj_head=true` | (openstroke29) Đầu lớp từng nét (6 lớp chung) |
| `feature_drop=layer_stats`, `restyle_mode=order` | Ẩn thống kê layer của trang; augment màu giữ thứ tự sáng tối |
| `viewer_share=0.65` | Tỉ lệ cửa sổ train lấy từ trang Viewer |
| `resume=<run>` | Chạy tiếp một run bị ngắt, giữ lịch LR / optimizer / EMA |

## Train cho một bộ dữ liệu mới

Kiến trúc, lệnh train và code suy luận giữ nguyên; chỉ dữ liệu và checkpoint là riêng.

1. **Dữ liệu:** trang GT, nhãn theo `relabel.object_strokes(<hệ>, ...)`, chia tập theo PDF / dự án, đóng băng trước khi train.
2. **Thẻ nghiên cứu** `docs/OPENSTROKE<N>.md` trước khi train (mục [Quy trình](#quy-trình-nghiên-cứu)).
3. **Train** như lệnh ví dụ, đổi `data=`, `modes=`, `name=`.
4. **Kiểm** theo mục dưới, rồi đặt checkpoint vào `models/`, suy luận với `mode=` đúng hệ đã train.

## Kiểm một model mới

Trước khi dùng một checkpoint làm nguyên liệu cho hậu kỳ / hệ thống:

1. **F2 nét trên val** (log train, `result.json`): tổng và **từng trang**; trang sập phải nằm trong báo cáo.
   **F2 object cũng nằm trong log** (`openstroke/object_eval.py`, user 2026-11-19: chỉ cần gửi log): mỗi epoch một dòng `objects` (GT / Viewer),
   cuối run bảng theo nhóm và từng trang + dòng `OBJECTS <strokes|pieces> <nhóm>: F2 P R`, lưu ở `result.json["objects"]`. Chỉ từ model: nét chọn
   (≥ ngưỡng checkpoint) nối theo cạnh "cùng object" ≥ 0,5, khớp IoU chiều dài ≥ 0,5, không lớp (đầu nét không có lớp); mảnh (OpenStroke34) có
   lớp trên trang GT. Không dùng logic AI_Takeoff, không cần score pack. `best.pt` vẫn chọn theo F2 nét.
2. **So với đối chứng cùng split, cùng protocol** (ví dụ E114). Số khác split không so được.
3. **`val_scores.npz` / `all_scores.npz`** chỉ cần gửi khi làm hậu kỳ có logic AI_Takeoff (chạy ở máy); đánh giá thường ngày đọc từ log.
4. **Kết quả suy luận khớp lúc train:** `Predictor` trên trang val phải cho cùng điểm với `val_scores.npz` (cùng TTA).
5. **Test** CPU qua (mục [Test](#test)).

## Quy trình nghiên cứu

Theo OpenOBB:

- Mỗi version kiến trúc có thẻ `docs/OPENSTROKE<N>.md` **trước khi train**: bằng chứng của version trước, mục tiêu, giả thuyết, kiến trúc và luồng
  dữ liệu, dữ liệu / split / metric, baseline và ablation, kỳ vọng định lượng ghi trước, tiêu chí đạt / bác bỏ, lệnh, kết quả thật, điểm yếu.
  Mẫu: [docs/RESEARCH_CARD_TEMPLATE.md](docs/RESEARCH_CARD_TEMPLATE.md).
- Sổ cái `research/LEDGER.md` (phát hiện F1, F2...), `research/HYPOTHESES.md` (giả thuyết H1, H2..., biến duy nhất được đổi, tiêu chí trước khi
  chạy) và script đo `research/probes/` **chỉ nằm ở máy nghiên cứu, không đưa lên git**. Thẻ trong `docs/` có thể nhắc mã F / H của sổ cái.
- Đặt tên run: `E<số>` tăng dần (E114, E115...). Kết quả thật chỉ lấy từ log train; chưa có số thì để trống.
- Phần mới của kiến trúc khởi tạo zero-init, có test "bắt đầu đúng bằng bản trước".
- Không train hay chạy suy luận dài trên CPU local; train trên Colab.
- License: chỉ thành phần MIT / Apache-2.0 / BSD; không code / weights Ultralytics; không import PyMuPDF trong package.

Trạng thái và nhật ký mới nhất: [AGENTS.md](AGENTS.md).

## Các kiến trúc

Thẻ mỗi version trong `docs/`. Tên kiến trúc là version nghiên cứu, lưu ở `config["arch"]` của checkpoint; checkpoint cũ lưu `openstroke2`
được nhận theo công tắc. **Các bảng dùng tập trang, nhãn và metric khác nhau, không so chéo.**

**Hướng nét không layer (hiện tại), HVAC, split `hvac10b`, 13 trang val, F2 nét (TTA ×8):**

| `model=` | Run | Ý chính | Tham số | F2 nét | Thẻ |
|---|---|---|---|---|---|
| `openstroke34` | E127 | openstroke31 + đầu object trên mảnh nét (cắt tại góc / chỗ chạm), lan độ tin cậy qua mảnh chạm / đối diện | 17,3M | chưa train | [OPENSTROKE34](docs/OPENSTROKE34.md) |
| `openstroke31` | E122 | openstroke30, loss đầu "cùng object" cân bằng trên cặp nét đúng | 17,21M | chưa train | [OPENSTROKE31](docs/OPENSTROKE31.md) |
| **`openstroke30`** | E120 | openstroke28 + attention theo đồ thị nối + đầu "cùng object" + loss bắt đủ object; nhãn cung Elbow / Tee | 17,21M | **0,857** | [OPENSTROKE30](docs/OPENSTROKE30.md) |
| `openstroke29` | E117 | openstroke28 + đầu lớp từng nét (6 lớp chung) | 17,13M | 0,842 | [OPENSTROKE29](docs/OPENSTROKE29.md) |
| `openstroke28` | E114 | Model nét không đọc layer + nhánh ảnh FPN | 17,13M | 0,839 | [OPENSTROKE28](docs/OPENSTROKE28.md) |
| (OpenStroke27) | E111 | Detector hộp (bản OpenOBB nhận nét), F2 object 0,643 | | | [OPENSTROKE27](docs/OPENSTROKE27.md) |
| (OpenStroke7–26) | | Ra thẳng object / instance: `obj_head`, detector OpenOBB + nét (`openstroke obb ...`) | | dưới E111 | `docs/OPENSTROKE7.md`… |

**Phân loại nét FIRE + HVAC, có đọc layer, kiểm chéo 5 fold, 31 trang GT, F2 nét:**

| `model=` / run | Ý chính | Tham số | F2 | FIRE | HVAC | Thẻ |
|---|---|---|---|---|---|---|
| OpenStroke5 (E37) + nét-dọc-nét | Nhánh ảnh FPN tới stride 4, khởi tạo `e32p` | 17,13M | 0,847 | 0,784 | 0,869 | [OPENSTROKE5](docs/OPENSTROKE5.md) |
| OpenStroke2 + ảnh (E30) | `img_lum`, `restyle_mode=order` | 8,46M | 0,818 | 0,708 | 0,860 | [OPENSTROKE2_IMAGE](docs/OPENSTROKE2_IMAGE.md) |
| OpenStroke2 + tiền huấn luyện (E26) | Tiền huấn luyện che nét | 5,66M | 0,781 | 0,684 | 0,820 | [OPENSTROKE2_IMAGE](docs/OPENSTROKE2_IMAGE.md) |
| `openstroke3` / `openstroke4` | Truyền tin dọc đồ thị nối / LitePT (tập dev 8 trang) | 5,5–5,7M | ≈ v2 | | | [OPENSTROKE3](docs/OPENSTROKE3.md), [OPENSTROKE4](docs/OPENSTROKE4.md) |
| `openstroke1` | kNN attention trên cửa sổ 4096 nét + LayerPool (fold cũ, nhãn e0) | 1,08M | 0,714 | | | [OPENSTROKE1](docs/OPENSTROKE1.md) |

## Kiến trúc

Kiến trúc hiện tại `openstroke30`: thân vector OpenStroke2 + nhánh ảnh FPN (thẻ [OPENSTROKE5](docs/OPENSTROKE5.md)), đọc đầu vào không layer
([OPENSTROKE28](docs/OPENSTROKE28.md)), thêm attention theo đồ thị nối và đầu "cùng object" ([OPENSTROKE30](docs/OPENSTROKE30.md)).

```
payload: shapes (items, màu, độ dày, nét đứt, fill)
  ├─ đặc trưng từng shape: 47 đặc trưng + 16 điểm lấy mẫu dọc nét (bezier lấy theo cung); layer coi như một (no_layers)
  ├─ nhánh vector: U-Net transformer kiểu Point Transformer V3 trên shape, cửa sổ 65.536 shape
  ├─ nhánh ảnh: cửa sổ vẽ lại thành ảnh từ chính các token → ResNet18 + FPN stride 4 → đặc trưng tại 16 điểm của nét, chiếu zero-init
  ├─ (openstroke30) 2 khối attention trên đồ thị nối: mỗi nét nhìn các nét chạm đầu mút / nối khe nét đứt / chữ T
  ├─ đầu nét → p(nét cần bóc) cho từng shape
  └─ (openstroke30) đầu cạnh → p(hai nét nối nhau cùng một object)
```

**Ảnh do model tự vẽ, không lấy từ PDF.** Mỗi shape vẽ thành đường gấp khúc qua 16 điểm của nó, 1 px / pt (cạnh ảnh tối đa 1536 px), mực đen
trên nền trắng; với `img_lum` mỗi nét mang độ đậm màu của nó. Ảnh nằm trong hệ toạ độ của cửa sổ, nên lật / quay / zoom khi augment áp
lên ảnh và vector như nhau, và ảnh khớp từng pixel với nét. Train, val và suy luận dùng cùng cách vẽ này, nên framework chỉ cần vector.
Ảnh chưa vẽ độ dày thật (nét rộng khoảng 1 px) và chưa tô vùng fill; độ dày vẫn vào model qua đặc trưng vector.

## Test

Chạy CPU, dữ liệu tổng hợp, không cần AI_Takeoff hay GPU:

```bash
OMP_NUM_THREADS=1 py -3.12 -m pytest -q tests
```

## Cấu trúc repo

| Thư mục | Nội dung |
|---|---|
| `openstroke/` | Thư viện: đọc payload, nhãn, đặc trưng, models, train (`engine/`), suy luận (`infer.py`), CLI |
| `openstroke/obb/` | Hướng cũ: detector OpenOBB nhận nét vector (CLI `openstroke obb ...`) |
| `app/` | App thử model HVAC (`hvac_app.py` + `hvac_viewer.html`), app Streamlit cũ |
| `tools/` | Trích payload, logic HVAC test, dựng dataset, điều khiển Colab / Kaggle |
| `colab/` | Notebook Colab |
| `docs/` | Thẻ từng version, mẫu thẻ, đánh giá vector, tốc độ train |
| `research/` | Sổ cái, giả thuyết, script đo: **chỉ ở máy nghiên cứu, không có trên git** (`.gitignore`) |
| `tests/` | `pytest` (CPU) |

Hậu kỳ dùng để test (`openstroke/postprocess.py`, `hvac_refine.py`, `hvac_faces.py`) đang nằm chung trong package; chúng không phải một phần của đầu ra raw.

## So với OpenOBB

Giống: bố cục repo, `pyproject.toml` + license, CLI kiểu YOLO (`key=value`), tên kiến trúc theo version nghiên cứu, checkpoint tự mang cấu hình,
app `run_app.bat`, thẻ version, phát hành bằng `tools/push_all.py` (version `0.<arch>.<patch>`, tag trên GitHub, bản sản phẩm trên GitLab).

Khác: sổ cái nghiên cứu không đưa lên git. Chưa có:
- Lớp API gọn kiểu `OpenOBB("best.pt").predict(...)` và lệnh `openstroke predict`; hiện dùng `openstroke.infer.Predictor`.
- Lệnh `strip` cho checkpoint giao đi (fp16, bỏ tham số train) và export ONNX.
- Kiểm định dạng payload đầu vào và mặt nạ "shape nào đã được chấm" trong đầu ra.

## Phát hành

```bash
python tools/push_all.py              # push GitHub + tag, rồi đồng bộ bản sản phẩm lên GitLab (đổi tên, chạy test, tag)
python tools/push_all.py --dry-run    # xem bản GitLab sẽ đổi gì, không đẩy
```

- `openstroke/__init__.py` `__version__` là version duy nhất; script tăng số patch khi package đổi kể từ lần phát hành trước.
- GitHub giữ mã, thẻ và notebook. GitLab nhận bản sản phẩm: package (không có `openstroke/obb`), test, app thử, thẻ của các kiến trúc đang dùng.

## License

Proprietary, xem [LICENSE](LICENSE). Thành phần bên thứ ba (MIT, BSD-3-Clause, Apache-2.0) và lưu ý về weights ImageNet: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
