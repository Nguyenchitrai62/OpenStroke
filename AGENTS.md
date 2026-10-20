# AGENTS.md: OpenStroke - model phân loại nét vector cho mạng ống MEP

> File bàn giao cho mọi coding agent. Đọc hết trước khi làm gì. Cập nhật mục 8 (trạng thái) và mục 9 (nhật ký) sau mỗi bước.
> Repo mới, tách khỏi OpenOBB (`F:\Source_code\NEW_architecture`) vì đây là bài toán khác: không detect box trên ảnh, mà
> **gán nhãn cho từng nét vector** của PDF. Kế thừa ý tưởng của OpenOBB và các kiến trúc SOTA, không import OpenOBB lúc chạy.
> Kế hoạch này được lập ngày 2026-10-20 trong một session AI_Takeoff; số liệu dưới đây đo ngày đó.

## 1. Mục tiêu

Một model, một logic hậu kỳ. Model đọc toàn bộ nét vector của một trang PDF và trả về cho **mỗi shape** đúng một trong hai
nhãn: `true` (nét thuộc mạng ống) hoặc `other` (mọi thứ còn lại). Logic HVAC/FIRE hiện có của AI_Takeoff nhận các nét `true`
và làm phần còn lại (nối nét đứt, topology, fitting, DN).

Model này thay **toàn bộ chuỗi** layout → chọn layer → crop theo axis → line model cho luồng mạng ống. Lý do: chuỗi đó sập cả
trang khi một cổng cứng sai. Bằng chứng trong AI_Takeoff:
- `0875-FP-CT2-C2Z1-303.pdf`: bước chọn layer FIRE lấy nhầm layer kiến trúc, F2 0,97 → 0,22.
- `7. FP_page 16.pdf`: box axis cắt mất 47,8 % chiều dài ống GT, F2 0,18 (0,75 khi bỏ crop).
- Classifier layer PLUMBING đổi model trên server mà không báo, cho `orther` toàn bộ.

Ngoài phạm vi (luồng khác, để nguyên): sprinkler, fan, soft pipe, thiết bị trong bảng symbol, Tee/Elbow FIRE (sinh từ topology),
ống nước HVAC (luồng water-pipe riêng của FIRE; **mặc định bỏ, user chưa chốt**), ELCV.
**PLUMBING:** từ 2026-10-20 user cho thêm GT PLUMBING làm dữ liệu train (nét ống nước giống ống FIRE; nhãn theo quy tắc FIRE, line
`Ống…`). Có cắm model vào luồng PLUMBING của AI_Takeoff hay không là việc sau, user chưa chốt.

## 2. Nhãn (đã chốt với user)

- **Đơn vị = một shape** trong payload `split_countfire` (`shapes[i]`, có thể chứa nhiều item `l`/`c`/`re`/`qu`). Không tách nhỏ.
  Nét đứt trong các PDF này đã bị CAD bung thành nhiều shape ngắn riêng (không trang nào dùng thuộc tính `dashes`): mỗi đoạn
  ngắn nằm trên ống là `true`.
- **FIRE:** shape `true` khi phần lớn chiều dài nằm dọc `Line` của GT (dung sai khoảng 2 pt).
- **HVAC:** shape `true` khi phần lớn chiều dài nằm **trong** polygon `Duct`, `Elbow*`, `Concentric Reducer*`,
  `Eccentric Reducer*`, `Chân rẽ*` của GT. Không dùng quy tắc "nằm trên viền": đo trên hai trang, phần lớn nét trong-mà-không-trên-viền
  thuộc chính các layer ống gió (`M-_MEVSEQFR_E0_N_` 6.206 shape...), do polygon GT chỉ xấp xỉ cung Elbow.
  **Sửa 2026-10-20 (user duyệt, nhãn v2, sổ cái F18 / F20):** shape trong polygon chỉ là `true` khi phần lớn chiều dài nằm trên viền
  polygon (≤ 2 pt) **hoặc** layer của nó có ≥ 50 % chiều dài nằm trong ống gió trên trang đó. Nét bộ môn khác bị ống gió đè (kiến trúc,
  kết cấu, bãi xe, nội thất, chữ ghi chú) là `other`. Code: `labels.refine_hvac`; dataset `datasets/e2`.
- **Mọi thứ khác là `other`.** Không có nhãn `ignore` (quyết định của user).
- Đường cong bezier phải lấy mẫu dọc cung khi đo, không dùng dây cung nối hai đầu.
- **Chỉ dùng object của group `primary` và `reference`.** Group `detected`, `water`, `detection` là bản debug (output AI chưa
  qua người sửa), xem `AI_Takeoff/BE/app/evaluate.py` `DEFAULT_GROUPS`. Dùng chúng là dạy model chép lỗi cũ.

## 3. Nguồn dữ liệu (chỉ đọc)

- DB: `F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3`, bảng `ground_truths` (`pages_json` → `objects` có `shape_name`,
  `kind`, `group`, `vertices` ở hệ `display_top_left`) và `documents` (`filename`, `sha256`, `stored_name`). Mở `mode=ro`.
- Payload nét: `F:\Source_code\AI_Takeoff\BE\data\drawings\document-NNNNNN\page-PPPP.vK.json.gz`, lấy bản `K` lớn nhất không
  có `.view.`. Đây đúng là input mà SDK HVAC/FIRE nhận. Shape có `items`, `type` (`s`/`fs`), `color`, `fill`, `width`, `dashes`,
  `closePath`, `layer`, `seqno`, `rect`.
- **Toạ độ:** payload của `pdf_etl.run_pipeline` đã ở hệ display top-left, trùng hệ với GT. Đã kiểm trên `H1_KS.pdf` (`/Rotate 270`,
  2384 × 1684): giữ nguyên toạ độ thì 30,05 % điểm mực rơi vào ống GT, xoay 90° hai chiều chỉ 13,94 % / 13,61 % (mức ngẫu nhiên).
  Bộ chuyển nhãn vẫn phải có chốt kiểm theo trang, kiểu `calibrate_rotation` trong `AI_Takeoff/BE/app/geometry.py`: chọn phép biến
  đổi có độ phủ mực cao nhất và từ chối trang nếu kết quả không rõ ràng.
- **Không bao giờ ghi** vào DB, cache hay bất kỳ file nào của AI_Takeoff. Ground Truth là input chỉ-đọc; chỉ người dùng sửa GT trên UI.
- Dữ liệu dẫn xuất (npz, manifest) nằm trong `datasets/` của repo này, gitignored. Không commit dữ liệu hay weights.
- Code model không import PyMuPDF (AGPL). Chỉ đọc payload JSON đã có.

### Kiểm kê GT (2026-10-20)

| | FIRE | HVAC |
|---|---|---|
| Trang có GT, sau khi bỏ trùng theo (sha256, trang) | 11 | 25 |
| PDF khác nhau | 10 | 16 (10 trang của `Y1_10-20.pdf` là một dự án) |
| Object trong phạm vi (primary) | `Line` 4.247 | `Duct` 3.171, Elbow 1.264, Reducer 740, Chân rẽ 717 |
| Shape mỗi trang | 50k–364k | 32k–812k |

Đo 6 trang (`research/probes/purity_probe.py`, chạy từ `AI_Takeoff/BE` bằng `.venv` của nó):

| Trang | Hệ | Shape | Shape 1 item | Shape `true` | Shape trộn 10–90 % |
|---|---|---|---|---|---|
| `7. FP_page 16.pdf` (doc 35) | FIRE | 243.547 | 91,9 % | 4,0 % | 337 |
| `02.DN-TT-FP-BLOCK A_R1-Model.pdf` (doc 62) | FIRE | 152.255 | 69,3 % | 7,3 % | 3.333 |
| `0875-FP-CT2-C2Z1-303.pdf` (doc 60) | FIRE | 246.327 | 99,3 % | 2,3 % | 182 |
| `20. 9062_M-ACB1----304.1` (doc 22) | HVAC | 105.127 | 89,3 % | 20,9 % | 458 |
| `H1_KS.pdf` (doc 43, xoay 270) | HVAC | 370.668 | 65,5 % | 25,4 % | 2.147 |
| `2D.pdf` (doc 51) | HVAC | 38.798 | 77,6 % | 42,3 % | 381 |

Shape trộn trên doc 62 phần lớn là nét chữ (`TAN THANH - FF- TEXT` 1.848) và kiến trúc đè lên ống: nhãn nguyên shape (`other`) là
đúng với chúng. 622 shape trộn trên chính layer ống `TAN THANH - FF` chưa xem.

### Cần user quyết (đừng tự quyết)

1. **Năm PDF có hai GT khác nhau trên cùng trang:** `333.pdf` (doc 20: 705 object, doc 33: 365), `NOBU [23].pdf` (26 / 45),
   `NOBU [24].pdf` (25 / 44), `MB TH4.pdf` (23 / 56), `2. TAHK_HAM-H-06.pdf` (54 / 55). User chọn bản dùng.
2. **GT phải phủ đủ cả trang** thì quy tắc "còn lại là other" mới đúng. Báo cáo QA liệt kê nét `other` giống ống để user xác nhận.
3. **Trang không phải mặt bằng làm mẫu âm** (sơ đồ nguyên lý, chi tiết, chú thích): hiện mọi trang GT đều là mặt bằng. Không có
   mẫu âm thì model sẽ đánh dấu ống của sơ đồ nguyên lý. User chọn trang và xác nhận "toàn bộ other".
4. **Chia fold theo PDF** (không theo tile, không để hai bản cùng PDF ở hai phía): user chốt manifest fold một lần. Agent không tự
   chuyển file giữa train và val, không loại file làm giảm điểm.
5. Ống nước HVAC có vào phạm vi không (mặc định không).
6. ~~Remote Git~~: **đã có** https://github.com/Nguyenchitrai62/OpenStroke (user, 2026-10-20). Tên package cuối cùng vẫn chờ
   (tên làm việc: OpenStroke).

## 4. Metric

- **Model:** recall / precision / F2 của nhãn `true` **theo chiều dài**, cộng tổng (micro) trên toàn bộ trang val của mọi fold.
  Ưu tiên recall: nét model bỏ thì logic hậu kỳ không lấy lại được.
- **Sản phẩm (giai đoạn sau, trong AI_Takeoff):** F2 của pipeline khi cắm mặt nạ vào, so với cascade hiện tại trên cùng trang.
  Ngưỡng của model chọn theo F2 pipeline, không theo accuracy.
- Báo cáo theo trang để thấy trang sập; thêm chốt "tỷ lệ nét `true` của trang lệch hẳn khỏi khoảng bình thường".

## 5. Baseline phải có trước model

1. **Trần của cách chọn theo layer:** với nhãn GT, chọn tập layer tối ưu cho từng trang và đo recall/precision theo chiều dài. Cho
   biết lọc theo layer tốt nhất có thể là bao nhiêu, và trang nào (không có layer thật, như doc 35) không cứu được bằng layer.
2. **Cascade hiện tại:** layer AI_Takeoff đã chọn trong run artifacts (`BE/data/runs/job-NNNNNN/`), đo bằng cùng metric.
3. **Model cây** (gradient boosting) trên đặc trưng tay của shape: độ dày, màu, fill, độ dài, hướng, thống kê của layer, mật độ lân
   cận. Nếu nó đã gần bằng transformer thì phần kiến trúc chưa đáng tiền.

## 6. Hướng kiến trúc (tóm tắt, chi tiết trong thẻ version)

- Token = một shape: loại item, điểm lấy mẫu dọc nét (gồm bezier), độ dày, màu nét/fill, `closePath`, `seqno`, id layer (chỉ để
  gộp nhóm, không có nghĩa xuyên bản vẽ), kích thước. Thêm token text (span chữ + bbox) nếu PDF có chữ nhúng.
- 39k–812k token mỗi trang: bắt buộc tile + attention thưa (window / serialization kiểu Point Transformer V3, hoặc kNN).
- Gộp theo layer: `LayerPool` của OpenOBB (H16, ý tưởng SymPoint-V2), mean + max theo layer trả về từng token, zero-init.
- Tuỳ chọn: đặc trưng ảnh lấy mẫu tại vị trí nét (họ VecFormer / CADTransformer; OpenOBB `vector.py` có cách splat ngược lại).
- Head nhị phân, loss có trọng số theo lớp hoặc chiều dài (FIRE chỉ 2–7 % shape là `true`).
- Thí nghiệm tokenize về sau (đổi đúng một biến): ghép các shape liền nhau theo `seqno`, cùng style và nối đầu-cuối thành một token.
  Phải đo độ thuần nhãn trước khi dùng.
- Mã tham khảo trong OpenOBB: `openobb/models/vector.py` (H7 VectorBranch, H16 LayerPool, H17 MemberHead),
  `tools/pdf_vectors.py`, `openobb/data/vectors.py`. Kết quả cũ trên FloorPlanCAD: nhánh vector +2,51 mAP50:95; bỏ vector thì cửa
  đơn −6,8, bay-window −11,4 (sổ cái OpenOBB F7, c1 so với c4).

## 7. Quy trình nghiên cứu (kế thừa OpenOBB)

- Sổ cái `research/LEDGER.md`: mọi thí nghiệm, phát hiện, lỗi, hướng bị bác bỏ, có số. `research/HYPOTHESES.md` ghi giả thuyết,
  **biến duy nhất** được đổi và tiêu chí đạt/bác bỏ **trước khi chạy**.
- Mỗi thí nghiệm có baseline cùng điều kiện. Không claim khi chưa có số. Không nới tiêu chí sau khi có số.
- Mỗi version kiến trúc có thẻ `docs/OPENSTROKE<N>.md`: bằng chứng → cơ chế → kỳ vọng ghi trước → điều chứng minh sai.
- Phần mới zero-init trên bản trước, có test "bắt đầu đúng bằng bản trước".
- GPU: không có GPU local. Train trên Colab như OpenOBB (`NEW_architecture/docs/COLAB.md`, `tools/cx.py`, `tools/job.py`); copy
  và chỉnh công cụ sang repo này thay vì gọi chéo. Torch CPU trên máy này phải chạy 1 luồng (`OMP_NUM_THREADS=1`).
- License: chỉ thành phần MIT / Apache-2.0 / BSD; không code / weights Ultralytics; không import PyMuPDF trong package.
- Secrets: không in, không commit, không đưa vào manifest.

## 8. Trạng thái và bước tiếp theo

- [x] Kế hoạch, phạm vi, quy tắc nhãn, kiểm toạ độ (2026-10-20).
- [x] **E0 dataset (2026-10-20):** `openstroke/` (sources → geometry → labels → build), 41 bản GT / 36 trang vật lý đạt chốt
  toạ độ, `datasets/e0/*.npz` + `manifest.json` (schema `openstroke.e0.v1`), QA `research/E0_QA.md`, sổ cái F1–F7.
  Lệnh: `py -3.12 -m openstroke build-e0` rồi `qa-e0` (Python 3.12 hệ thống có numpy / shapely / torch CPU; không cần venv
  AI_Takeoff, không ghi gì vào AI_Takeoff).
- [ ] **User quyết mục 3** (đã hỏi 2026-10-20): cặp trùng (`duplicate_choice`), fold (`datasets/e0/folds_proposal.json`,
  `status: proposal`), mẫu âm, ống nước HVAC, GT phủ đủ trang cho các trang F4, remote Git.
- [x] E1.1 trần chọn layer (local, fold đề xuất, 31 trang không trùng): F2 micro 0,732 (R 0,875 P 0,443); doc 35 / 36 không layer
  0,305 / 0,147. Lệnh `py -3.12 -m openstroke baseline-layer`.
- [x] E2 model v1 hạt giống A, 5 fold (log user, trainer cũ): micro F2 0,714 (R 0,783 P 0,529), dưới trần layer 0,732; H3 bác bỏ;
  fold 4 0,599. Phân tích: sổ cái F11–F14, probe `research/probes/style_probe.py`.
- [ ] E1.3 cây (+ hạt giống B của v1 nếu cần) (thẻ `docs/OPENSTROKE1.md`, tiêu chí `research/HYPOTHESES.md` H1–H5): code + test xong, chờ user train
  trên Colab bằng `colab/OpenStroke_train_quick.ipynb` (khuôn `OpenOBB_train_quick.ipynb`): notebook cài code từ GitHub (`pip install git+https://github.com/Nguyenchitrai62/OpenStroke.git`),
  data từ `datasets/openstroke_e0_colab.zip` trên Drive (`py -3.12 -m openstroke pack`, chỉ data). **Push trước khi bảo user
  train**: Colab chạy đúng bản trên `main`. Notebook sinh bởi `tools/make_notebook.py`. **Không train trên CPU local** (user, 2026-10-20): chỉ chạy test.
- [ ] **E3 model v2** (`docs/OPENSTROKE2.md`, H6–H10): U-Net transformer kiểu PTv3 trên shape, cửa sổ 65536, khung train
  `openstroke/engine/` (kiểu YOLO / OpenOBB: EMA, warmup + cosine `lr0`/`lrf`, AMP bf16, resume, patience, results.csv/png, TTA).
  Code + test xong; chờ user chạy `openstroke bench` rồi ba hạt giống A / C / B trên Colab (thứ tự đó) và GBDT một lần (ô 6 notebook).
- [ ] **E4 PLUMBING** (`research/HYPOTHESES.md` H12–H14): dataset `datasets/e1` (FIRE + HVAC + 195 trang PLUMBING CTN), fold
  `datasets/e1/folds_plumbing.json` (fold 5 = cả dự án CTN), payload thiếu tạo bằng `tools/make_payloads.py` (venv AI_Takeoff, ghi vào
  `datasets/payloads/`). **Đã build** (236 / 236 trang đạt, 41 bản FIRE / HVAC trùng nhãn e0; sổ cái F16), zip
  `datasets/openstroke_e1_colab.zip`. Chờ user upload zip lên Drive và chạy **PM → PMN → P** (ô 5b notebook) sau khi A xong.
  PMN thêm `layer_names=true` (H15, sổ cái F19).
- [x] Quy tắc nhãn HVAC v2 (user duyệt): dataset **`datasets/e2`** (236 / 236 trang, FIRE / PLUMBING trùng e1, HVAC bỏ 14,9 % chiều
  dài true), zip `datasets/openstroke_e2_colab.zip`. **Mọi run từ nay trên e2**, thứ tự A2 → PMN → PM → P (notebook ô 5).
- [x] **App thử model** (`app/streamlit_app.py`, `streamlit run app/streamlit_app.py`): thả PDF → `tools/extract_payload.py` (venv
  AI_Takeoff, `pdf_etl`) → `openstroke/infer.py` (`Predictor`: cùng mảng / đặc trưng như trang train) → viewer `app/viewer/index.html`
  (ngưỡng, tô theo xác suất, bấm nét, tải mask JSON chỉ số shape). Đổi model: sửa `MODEL_PATH` hoặc gõ trong panel.
- [ ] E1.2 cascade hiện tại từ run artifacts AI_Takeoff.
- [ ] Sau đó: đường cắm mặt nạ vào AI_Takeoff (mức thô: mặt nạ → `layers_by_page`; mức đầy đủ: lọc shape trước SDK và tắt crop,
  cần kiểm `Extract_FIRE` có đọc lại PDF theo OCG khi render ảnh layer không). Việc đó làm trong AI_Takeoff theo luật của nó,
  không trong repo này.

## 9. Nhật ký

- 2026-10-20: Tạo repo. Kế hoạch từ session AI_Takeoff; probe đo ở `research/probes/purity_probe.py`.
- 2026-10-20: E0 xong. Chốt toạ độ dùng mực vector của payload (không PyMuPDF), dung sai 0,5 pt, điểm ngoài trang tính trượt
  (sổ cái F2, F3: hai lỗi của cách chấm kiểu AI_Takeoff). Thêm E1.1 (local), E1.3 GBDT + model v1 + notebook Colab theo yêu cầu
  user; fold chỉ là đề xuất, 5 trang GT trùng bị bỏ qua cho tới khi user chọn. Test: `OMP_NUM_THREADS=1 py -3.12 -m pytest -q tests`
  (28 test, khoảng 10 s).
- 2026-10-20: User tạo remote GitHub và push commit `init`. Notebook chuyển sang cài code từ GitHub; zip Colab chỉ còn data.
- 2026-10-20: CLI nhận cú pháp kiểu YOLO / OpenOBB (`openstroke train data=... model=openstroke1 epochs=30 layer_pool=false`),
  song song với `--flag`. `model=<thư mục run>` fine-tune mỗi fold từ `fold<k>/best.pt` của chính nó (không rò val);
  `model=<một .pt>` chạy được nhưng log cảnh báo rò val. Run tự đánh số `runs/train`, `train2`...
- 2026-10-20: Log v1 của user (fold 0): F2 0,810 ở epoch 5 rồi giảm → overfit; VRAM 5 / 96 GB. Đo ngữ cảnh (F8) và pooling (F9),
  viết v2 + khung train mới (`openstroke/engine/`, `data.py`, `serialize.py`, `models/`), lệnh `openstroke bench`. Mặc định
  `model=openstroke2`; v1 vẫn chạy qua `model=openstroke1` và nạp được checkpoint cũ. Test: 40 (khoảng 20 s).
- 2026-10-20: Log v1 đủ 5 fold (user; chạy bằng trainer trước `ed0c90c`). Micro F2 0,714, best ở epoch 5–20, fold 4 0,599. Probe style
  (bảng tra, không học): style ống của fold 3 / 4 chưa gặp trong train ở 51 % / 33 % chiều dài true; bảng tra 0,33–0,47. Model chưa
  biết mode FIRE / HVAC (F14). Bước kế: v2 A / B trên Colab sau khi chạy lại ô cài (lấy `main`).
- 2026-10-20: Augment `restyle` (mỗi màu / độ dày của cửa sổ → giá trị ngẫu nhiên, giữ nhóm; `data.restyle`), mặc định 0,5 cho v2,
  0 cho v1; hạt giống C = `restyle=0` (H11, ghi trước). Notebook: thứ tự A → C → B, thêm ô GBDT (CPU) và gom `runs/e1_gbdt` vào zip.
  Test: 42.
- 2026-10-20: Lần chạy v2 đầu trên Colab không học (BCE kẹt ln 2, đoán `true` mọi shape). Hai lỗi: khởi tạo mặc định (logit như nhau
  mọi shape) và Lovász hinge kéo ngược BCE. Sửa: xavier + bias 0, Lovász trên xác suất; kiểm 60 bước CPU trên trang thật. Test: 43.
  **Luật mới:** trước khi gửi kiến trúc mới lên Colab, chạy vài chục bước trên trang thật ở CPU, loss phải rời ln 2.
- 2026-10-20: v2 sau sửa học được (fold 0 ep 9: F2 0,821, F2@0,5 0,800). Thêm `parallel=N`: mỗi fold một tiến trình con trên cùng
  GPU (`--child`, log `fold<k>/train.log`), tiến trình chính in dòng val từng fold và gộp summary; kết quả trùng chạy tuần tự (test).
  Notebook mặc định `parallel=5`. Test: 44.
- 2026-10-20: Thêm PLUMBING (remote `aitakeoff.leductu.online`): 195 trang CTN, payload tạo bằng venv AI_Takeoff, dataset `e1`, fold 5,
  `plumbing_share`, `mode_input`, mốc đoán-tất-cả in trong log. Lượt sync remote đã đổi ID tài liệu và xoá cache cũ: build lại từ
  AI_Takeoff giờ cần `tools/make_payloads.py` trước. Resume khôi phục đủ tham số của run cũ. Test: 48.
- 2026-10-20: App Streamlit thử model (kiểu OpenOBB). Kiểm: đặc trưng từ payload trùng tuyệt đối đặc trưng trang train (2 trang thật);
  `Predictor` với `openstroke_v2.pt` của user (run A fold 0, epoch 50) trên 2 trang val của fold 0 ra micro F2 0,8436 so với 0,8435
  trainer báo. CPU 1 luồng: trang 152k shape ~22 s trích + 6 s đặc trưng + 42 s model. Test: 49.
- 2026-10-20: Phân tích lỗi checkpoint fold 0 (`research/probes/error_probe.py`): FP FIRE là ống của hệ khác (thoát nước, hút khói;
  F17); FN HVAC phần lớn là nét kiến trúc bị quy tắc polygon gán true (F18, chờ user quyết). Probe tên layer (F19): tên có nghĩa xuyên
  bản vẽ ở 2 / 5 fold. Thêm đầu vào `layer_names` (zero-init, `name_drop`), thứ tự E4 thành PM → PMN → P, notebook 3 ô chạy nguyên.
  Test: 53.
- 2026-10-20: User duyệt quy tắc HVAC v2 và giao toàn quyền nghiên cứu (mục tiêu: 1 model + 1 logic hậu kỳ, sát GT nhất). Build
  `e2`: FIRE / PLUMBING trùng e1; HVAC −14,9 % chiều dài true (kiến trúc, xe, nội thất, chữ). Mốc đoán-tất-cả fold 0–4 của e2:
  0,574 / 0,401 / 0,267 / 0,292 / 0,451; fold 5 0,133. Notebook chuyển sang e2. Test: 54.
