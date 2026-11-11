# OpenStroke

**HVAC vector không layer — OpenStroke12:** [thẻ nghiên cứu](docs/OPENSTROKE12.md), [notebook E90/E91](colab/OpenStroke12_E91.ipynb). Một xác suất HVAC chung train/infer, soft-F2 chiều dài, best theo F2 object toàn trang. Chưa có kết quả GPU v12. App đọc ngưỡng từ checkpoint; chạy `run_app.bat "D:\Models\openstroke12.pt"`.

Phân loại từng nét vector của PDF bản vẽ MEP: nét thuộc mạng ống (`true`) hay mọi thứ còn lại (`other`). Model đọc toàn bộ nét của một trang
và trả về một xác suất cho mỗi shape, không qua bước chọn layer hay crop nào có thể làm sập cả trang. Cấu hình tốt nhất: **OpenStroke5** (`img_fpn=true`).

> **Trạng thái: nghiên cứu, chưa có số so sánh cuối.** Mục tiêu bắt buộc là model + logic hậu kỳ chính xác hơn pipeline AI_Takeoff hiện tại trên cùng trang.
> Số của model dưới đây mới là mặt nạ nét, **chưa qua** logic FIRE / HVAC; số của pipeline đã qua toàn bộ logic đó. Chưa cắm vào AI_Takeoff.

## Kết quả

Kiểm chéo 5 fold theo PDF, 31 trang GT (FIRE + HVAC). F2 / recall / precision của nhãn `true` **theo chiều dài**, cộng tổng trên mọi trang val.
Có hậu kỳ nét-dọc-nét (mục [Hậu kỳ](#hậu-kỳ)), ngưỡng theo hệ.

| | Tham số | F2 | R | P | FIRE | HVAC |
|---|---|---|---|---|---|---|
| **OpenStroke5 (E37) + hậu kỳ** | 17,13M | **0,847** | 0,912 | 0,660 | 0,784 | 0,869 |
| OpenStroke2 + ảnh (E30) + hậu kỳ | 8,46M | 0,837 | 0,901 | 0,651 | 0,762 | 0,862 |
| Pipeline AI_Takeoff, cùng trang, **đã qua toàn bộ logic** | — | ~0,957 | ~0,969 | ~0,911 | ~0,937 | ~0,964 |

Đọc số này cho đúng:

- **Hai dòng trên và dòng pipeline chưa cùng giai đoạn.** Model là đầu vào của logic hậu kỳ, nên precision thấp ở đây chưa phải kết quả cuối: logic còn lọc tiếp. Riêng bước chọn layer của pipeline
  chỉ giữ 0,790 ở độ thuần 0,498 (29 trang), rồi logic sau đó mới đưa precision lên ~0,91. So sánh đúng là cho mặt nạ của model đi qua cùng logic đó, việc này chưa làm.
- **Pipeline là số trong mẫu, model là số ngoài mẫu.** 71 % object GT FIRE / HVAC sinh từ output AI, và code pipeline được tối ưu trên chính các trang này,
  nên số của pipeline lạc quan (sổ cái F29, F38). Nó được tính lại từ R, P đã làm tròn của lần chạy 2026-10-21.
- **Số của model cũng hơi lạc quan:** ngưỡng hậu kỳ chọn trên chính trang val; fold vẫn ở trạng thái đề xuất, chưa được chủ dữ liệu chốt.
- **Train có dùng nhãn giả:** ngoài GT, mỗi fold train thêm 29 trang nhãn giả từ pipeline. Val chỉ dùng GT.
- **Chưa phủ:** PLUMBING không nằm trong các số này; 5 PDF có hai bản GT khác nhau đang bị bỏ qua.
- E37 so với E30 đổi hai thứ cùng lúc (khởi tạo và FPN), xem [thẻ](docs/OPENSTROKE5.md).

Khi logic lọc tiếp, cái cần giữ là recall, vì nét model bỏ thì logic không lấy lại được. Điểm E37 trên 31 trang, chưa có hậu kỳ nét-dọc-nét:

| Ngưỡng | R | P | R FIRE | R HVAC |
|---|---|---|---|---|
| 0,20 | 0,901 | 0,625 | 0,858 | 0,917 |
| 0,10 | 0,939 | 0,558 | 0,881 | 0,960 |
| 0,05 | 0,956 | 0,491 | 0,893 | 0,979 |
| 0,02 | 0,968 | 0,403 | 0,904 | 0,990 |

**Cùng logic HVAC, 20 / 21 trang HVAC mà bản chạy lại tái hiện được pipeline** (`research/probes/logic_hvac_probe.py`, không có detector, symbol; chưa có FIRE): pipeline thật 0,956, đối chứng layer của pipeline 0,946, **E37 chọn layer + nét tự tin 0,941** (R 0,949, P 0,912). Không trang nào E37 hơn đối chứng quá 0,01.

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

Cần Python ≥ 3.10. Tuỳ chọn thêm: `openstroke[app]` (app thử model), `openstroke[baseline]` (`baseline-gbdt`), `openstroke[dev]` (pytest).

## Dữ liệu

Dữ liệu và weights **không nằm trong repo**. Dataset dựng từ DB và payload của AI_Takeoff, mở chỉ-đọc, nên chỉ dựng lại được trên máy có AI_Takeoff.

```bash
py -3.12 -m openstroke build-e0 --out datasets/e2      # GT + payload -> nhãn theo shape
py -3.12 -m openstroke qa-e0 --manifest datasets/e2/manifest.json    # báo cáo QA
py -3.12 -m openstroke build-pseudo                    # trang nhãn giả từ output pipeline -> datasets/pseudo1
py -3.12 -m openstroke build-unlabeled                 # trang chưa nhãn, chỉ để tiền huấn luyện
py -3.12 -m openstroke merge-data --parts datasets/e2 datasets/pseudo1 --out datasets/e3
py -3.12 -m openstroke pack --data datasets/e2 --folds datasets/e2/folds.json --out datasets/openstroke_e2_colab.zip
```

- Quy tắc nhãn. **FIRE:** `true` khi phần lớn chiều dài nằm dọc `Line` của GT (≤ 2 pt). **HVAC:** `true` khi phần lớn chiều dài nằm trong polygon ống gió, đồng thời nằm trên viền polygon
  hoặc thuộc layer có ≥ 50 % chiều dài trong ống gió. Chi tiết: [AGENTS.md](AGENTS.md) mục 2.
- Code package chỉ đọc payload JSON. Trích PDF thành payload (`tools/extract_payload.py`, `tools/make_payloads.py`) chạy bằng Python của AI_Takeoff, có `pdf_etl`.

## Train

Train trên GPU; repo này chạy trên Colab G4 (`colab/OpenStroke_train_quick.ipynb`, `tools/colab_run.py`). Cú pháp giống YOLO. Bắt buộc có `model=`: tên kiến trúc,
hoặc một `best.pt` để fine-tune, hoặc thư mục run để mỗi fold fine-tune từ `fold<k>/best.pt` của chính nó (không rò val).

Cấu hình tốt nhất gồm hai bước: tiền huấn luyện che nét trên trang không thuộc PDF có GT, rồi fine-tune có giám sát (`<D>` = thư mục data).

```bash
openstroke train model=openstroke2 data=<D> folds=<D>/folds_pretrain_clean.json fold=0 modes=fire,hvac feature_drop=layer_stats batch=4 lr0=0.002 image_branch=true img_lum=true restyle_mode=order pretrain=0.4 epochs=40 tta=false name=e32p
openstroke train model=runs/e32p/fold0/best.pt data=<D> folds=<D>/folds_cv_pseudo.json fold=0,1,2,3,4 modes=fire,hvac feature_drop=layer_stats batch=4 lr0=0.002 image_branch=true img_lum=true restyle_mode=order layer_head=true epochs=15 img_fpn=true name=e37
```

Mỗi fold ghi `best.pt`, `result.json`, `val_scores.npz`, `results.csv`, `results.png` vào `runs/<name>/fold<k>/`. E37 dùng 31,6 GB VRAM, 1,8 it/s trên G4.

| Cờ | Tác dụng |
|---|---|
| `image_branch=true` | Vẽ cửa sổ ra ảnh từ chính vector, ResNet18 pretrained, đặc trưng lấy tại điểm của nét |
| `img_fpn=true` | Thêm layer4 và FPN tới stride 4 cho nhánh ảnh (OpenStroke5) |
| `img_lum=true`, `restyle_mode=order` | Giữ thứ tự độ đậm của nét, ở ảnh và ở augment |
| `layer_head=true` | Thêm đầu chọn layer cạnh đầu nét |
| `feature_drop=layer_stats` | Ẩn thống kê layer của trang (lối tắt làm sập trang lạ, F28) |
| `pretrain=0.4` | Tiền huấn luyện che 40 % nét, không cần nhãn |
| `resume=<run>` | Chạy tiếp một run bị ngắt |

Test (chạy CPU, dữ liệu tổng hợp): `OMP_NUM_THREADS=1 py -3.12 -m pytest -q tests`.

## Suy luận

```python
from openstroke.infer import Predictor, load_payload

pred = Predictor("runs/e37/fold0/best.pt")
out = pred.predict_payload(load_payload("page-0001.v0.json.gz"), mode="fire")   # hoặc "hvac"
out["prob"]                 # (số shape,), đúng thứ tự payload; shape không có hình học = 0
mask = out["prob"] >= pred.threshold
```

`Predictor` trả điểm của model, **chưa có hậu kỳ**; hậu kỳ theo từng hệ nằm ở `openstroke/postprocess.py` và `tools/run_hvac_logic.py` (app dùng cả hai). `pred.threshold` là ngưỡng chọn trên val của fold đó, nên chỉ dùng để thử. Một `best.pt` chỉ "chưa thấy" các PDF trong val của fold đó.

**App thử model:** `run_app.bat` (Streamlit, cần `openstroke[app]`): thả PDF vào, bấm chọn trang trong danh sách có ảnh thu nhỏ (PDF một trang chạy ngay), xem tiến độ và thời gian từng bước, rồi xem kết quả sau hậu kỳ hoặc chỉ model theo ngưỡng, và tải mask. Sửa `MODEL=` ở đầu file. Mở PDF cần `OPENSTROKE_EXTRACTOR_PYTHON` trỏ tới Python có `pdf_etl`.

## Hậu kỳ

**Nét-dọc-nét** (F41), một luật, không tham số học: nhãn GT là quan hệ hình học với đường ống, nên mọi nét nằm dọc (≤ 2 pt, phần lớn chiều dài) các nét có p ≥ t cũng thành `true`.
Ngưỡng `t` chọn theo hệ: FIRE 0,55, HVAC 0,35 ở E37. Code: `openstroke.postprocess.along_confident`. Chấm trên kết quả val:

```bash
py -3.12 research/probes/relabel_probe.py e37     # đọc runs_colab/e37 và datasets/e3
```

## Các kiến trúc

Luồng HVAC không layer, ra object từ nét: [OpenStroke10](docs/OPENSTROKE10.md) và [OpenStroke11](docs/OPENSTROKE11.md) (E86, cổng HVAC trước k-NN).

Thẻ mỗi kiến trúc có bằng chứng, cơ chế, kỳ vọng ghi trước, lệnh và kết quả thật. Số chỉ lấy từ log của các lần train; bản nào chưa có số thì để trống đến khi một lần nghiên cứu sau đo.

**Kiểm chéo 5 fold**, 31 trang GT, cùng fold. Ngưỡng theo từng fold, không hậu kỳ; pipeline ~0,957.

| Cấu hình | Thêm vào dòng dưới | Tham số | F2 | R | P | FIRE | HVAC | Thẻ |
|---|---|---|---|---|---|---|---|---|
| **OpenStroke5 (E37)** | FPN tới stride 4, khởi tạo `e32p` | 17,13M | **0,841** | 0,919 | 0,628 | 0,747 | 0,876 | [OPENSTROKE5](docs/OPENSTROKE5.md) |
| OpenStroke2 + độ đậm (E30) | `img_lum`, `restyle_mode=order` | 8,46M | 0,818 | 0,900 | 0,599 | 0,708 | 0,860 | [OPENSTROKE2_IMAGE](docs/OPENSTROKE2_IMAGE.md) |
| OpenStroke2 + ảnh (E28) | nhánh ảnh ResNet18 | 8,46M | 0,809 | 0,891 | 0,593 | 0,735 | 0,837 | [OPENSTROKE2_IMAGE](docs/OPENSTROKE2_IMAGE.md) |
| OpenStroke2 + tiền huấn luyện (E26) | tiền huấn luyện che nét | 5,66M | 0,781 | 0,874 | 0,548 | 0,684 | 0,820 | [OPENSTROKE2_IMAGE](docs/OPENSTROKE2_IMAGE.md) |
| OpenStroke2 + đầu layer (E24) | đầu layer, nhãn giả | 5,65M | 0,751 | 0,839 | 0,529 | 0,669 | 0,784 | [OPENSTROKE2_IMAGE](docs/OPENSTROKE2_IMAGE.md) |

**Tập dev** 8 trang, **không so được với bảng trên**. Có TTA, không hậu kỳ; pipeline 0,947.

| `model=` | Kiến trúc | Tham số | Run | F2 | Thẻ |
|---|---|---|---|---|---|
| `openstroke3` | v2 + truyền tin dọc đồ thị nối của nét | 5,69M | E14, có nhãn giả | 0,677 (mốc cùng điều kiện 0,668, hoà) | [OPENSTROKE3](docs/OPENSTROKE3.md) |
| `openstroke2` | U-Net transformer tuần tự hoá kiểu PTv3, cửa sổ 65536 nét | 5,65M | D0-nl (`feature_drop=layer_stats`) / D0 | 0,625 / 0,580 | [OPENSTROKE2](docs/OPENSTROKE2.md) |
| `openstroke4` | LitePT: đồ thị ở tầng 0, conv ở tầng 1, attention từ tầng 2 | 5,53M | E6 | 0,585 trước TTA (D0 0,575), chậm hơn v2 1,7× | [OPENSTROKE4](docs/OPENSTROKE4.md) |

**Fold đề xuất cũ**, nhãn e0, không so được với hai bảng trên:

| `model=` | Kiến trúc | Tham số | F2 | R | P | Thẻ |
|---|---|---|---|---|---|---|
| `openstroke1` | kNN attention trên cửa sổ 4096 nét + LayerPool | 1,08M | 0,714 | 0,783 | 0,529 | [OPENSTROKE1](docs/OPENSTROKE1.md) |

## Kiến trúc OpenStroke5

Trang được cắt thành cửa sổ 65536 nét. Mỗi nét là một token: 47 đặc trưng (đã ẩn 6 đặc trưng thống kê layer) cộng PointNet trên 16 điểm lấy mẫu dọc nét.

```
payload -> nét -> token ---------------(+)--> U-Net transformer 5 tầng -> đầu nét   -> xác suất mỗi nét -> hậu kỳ nét-dọc-nét
             |                          ^     (PTv3, RoPE, LayerPool)  -> đầu layer
             v                          |
          vẽ ra ảnh -> ResNet18 + FPN tới stride 4 -> lấy đặc trưng tại 16 điểm của nét -> chiếu zero-init
```

Nhánh mới bắt đầu đúng bằng bản trước nhờ phép chiếu zero-init (có test). Chi tiết: [docs/OPENSTROKE5.md](docs/OPENSTROKE5.md).

## Cấu trúc repo

| Thư mục | Nội dung |
|---|---|
| `openstroke/` | Thư viện: dữ liệu, nhãn, đặc trưng, models, train, suy luận, CLI |
| `tests/` | `pytest` (CPU, dữ liệu tổng hợp) |
| `colab/` | Notebook Colab |
| `docs/` | Thẻ kiến trúc |
| `research/` | Sổ cái `LEDGER.md`, giả thuyết `HYPOTHESES.md`, script đo `probes/` |
| `tools/` | Trích payload, tạo fold, điều khiển Colab |
| `app/` | App Streamlit để thử model |

Quy trình nghiên cứu và trạng thái mới nhất: [AGENTS.md](AGENTS.md).

## License

Proprietary, xem [LICENSE](LICENSE). Thành phần bên thứ ba (MIT, BSD-3-Clause, Apache-2.0) và lưu ý về weights ImageNet: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
