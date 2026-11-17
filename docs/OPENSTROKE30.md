# OpenStroke30 — bắt đủ object: attention theo đồ thị nối đầu mút, đầu "cùng object", loss closure, nhãn cung cong

Ngày: 2026-11-17. Trạng thái: đã code và test CPU, chưa train. Sổ cái F165, giả thuyết H132–H134.

## 1. Mục tiêu và phạm vi

Model nét là một mắt xích của pipeline (user 2026-11-17). Model ưu tiên R, thà bắt thừa còn hơn bỏ sót; hậu kỳ dựng object từ khối khép kín và bỏ
nét lẻ không khép được khối nào. Điểm cuối là F2 object của pipeline, không phải F2 nét. Mục tiêu chung vẫn là F2 object ≥ 0,80 trên GT giữ ngoài train.

Version này nhắm ba lỗi user chỉ ra: miss nét cong (`c`), object 4 nét bắt được 3, và nhiễu thành khối kín giả. Input không layer như E114
(`openstroke-v1`). Không đổi split, nguồn GT, cách chấm.

## 2. Bằng chứng (0 GPU, điểm `last.pt` E116, 13 trang val hvac10b, TTA ×1)

**Nét cong không phải lỗi model mà là lỗi nhãn.** Nét cong nhãn đúng: R 0,995 (GT) / 0,996 (Viewer). Nét cong nằm sát viền Elbow / Tee /
Chân rẽ của GT (≤ 3 pt) nhưng nhãn `other`: GT 1.625 pt (nhãn đúng 6.303 pt), Viewer 3.495 pt (nhãn đúng 2.558 pt); model bắt 15 % / 2,7 % số
đó, tức học đúng theo nhãn. Lý do: nhãn v3 cần ≥ 50 % chiều dài cách cạnh GT ≤ 2 pt và lệch ≤ 20°, mà polygon GT vẽ cung bằng vài dây cung.
Trên cả 203 trang: nét cong `other` có hai đầu chạm nét đúng = 19,1 % chiều dài nét cong đúng (Viewer), 5,5 % (GT). Xem ảnh: có cả cung ngoài
Elbow bị bỏ (lỗi thật) lẫn ống mềm, cánh cửa, đám mây sửa đổi (đúng là `other`).

**Object thiếu nét.** 21 % object GT (GT5) chỉ được bắt một phần; 70 % chiều dài nét bị miss trên GT nằm trong các object đó; 48 % chiều dài
nét miss có đầu mút chạm một nét đúng đã bắt. Viewer: 18,6 % object bắt một phần, 14,5 % không bắt nét nào.

**Nhiễu.** Nét bắt nhầm: 41 % chiều dài không chạm nét bắt nào (GT; hậu kỳ tự bỏ), 42 % chạm một nét đúng (Viewer). Duct thừa ngoài GT
(F164): ô đỗ xe, dãy tủ, thiết bị — khối kín giả mà hậu kỳ không phân biệt được.

**Trần của model nét, F2 object** (GT5 / Viewer 8, rule_nl 6 lớp, conf 0,5):

| Nét | Hậu kỳ đầy đủ (logic + vùng + bắt buộc phủ, mặc định app) | Hậu kỳ chỉ khối khép kín |
|---|---|---|
| Model hiện tại (E116 `last.pt`) | 0,761 / 0,539 | 0,808 / 0,529 |
| Bắt đủ object (oracle: nét đúng của object đã bắt ≥ 1 nét) | 0,745 / 0,554 | 0,815 / 0,540 |
| Bỏ hết nhiễu (oracle: p = 0 cho nét `other`) | 0,785 / 0,594 | 0,826 / 0,574 |
| Nét hoàn hảo (oracle: p = nhãn) | 0,779 / 0,662 | **0,854 / 0,640** |

- Model nét tốt hơn có trần +0,05 (GT5) / +0,11 (Viewer) với hậu kỳ khối khép kín; đạt luật F107 (trần ≥ +0,10 trên Viewer, biết loại nút thắt).
- Trên trang GT train (cross-fit theo tài liệu), hậu kỳ chỉ khối khép kín 0,734 so với đầy đủ 0,796: `doc000196` 0,432 so với 0,870 vì ống vẽ
  bằng **nét đứt**, model bắt đủ đoạn nhưng khe giữa các đoạn không khép khối. Đó là việc của hậu kỳ (nối khe nét đứt), không phải model; ghi
  ở §12.

**Vì sao khối quan hệ cũ không giúp** (E14, OpenStroke3 `relations=2`, hoà trong nhiễu; OpenStroke4 `GraphMixer` chậm và sập): trung bình
tin nhắn MLP, và loss vẫn chấm từng nét riêng lẻ. Không có gì buộc model bắt **đủ** object hay học quan hệ thuộc cùng object.

## 3. Giả thuyết đăng ký trước

- **H134 (E118, dữ liệu):** nhãn cung cong khép object (e10) cho model học bắt các cung đó; bắt đủ cung giúp khối khép kín thành hình.
  Biến: e9g → e10. Đối chứng E117c (cùng công thức trên e9g).
- **H133 (E119, loss):** loss closure (soft-min logit các nét của một object GT → 1) làm model bắt đủ object. Biến: `closure=0.5`. Đối chứng E118.
- **H132 (E120, kiến trúc):** attention theo đồ thị nối (2 khối) + đầu "cùng object" cho mỗi cặp nét nối nhau cho model dùng nét bên cạnh
  để quyết nét này (một cạnh của Elbow kéo cạnh còn lại) và phân biệt khối kín thật / giả theo nét nối với nó. Biến: `rel_attn=2 edge_head=true`.
  Đối chứng E119.

## 4. Kiến trúc và luồng dữ liệu

```
nét (lệnh vẽ, màu, độ dày, điểm mẫu)  + ảnh vẽ lại (ResNet18 + FPN, E114)
  -> U-Net transformer cửa sổ (OpenStroke2, không đổi)
  -> [mới] GraphAttention × rel_attn trên đồ thị nối (graph.py: đầu mút chạm 1 / khe nét đứt 2 / chữ T 4)
       mỗi nét i: α_ij = softmax_j( q_i·k_j / √d + b(loại cạnh, hình học) ), h_i += W_o Σ α_ij (v_j + e_ij);  FFN; zero-init W_o, FFN cuối
  -> đầu nhị phân (giữ)                           -> logit nét
  -> [mới] EdgeHead( h_i + h_j, |h_i − h_j|, e_ij ) -> logit "cùng object" mỗi cạnh nối
```

- Hình học cạnh: cos / |sin| góc giữa hai nét, log khoảng cách tâm, log tỉ lệ độ dài (như RelationBlock).
- Loss: BCE + Lovász (giữ) + `closure` × closure_loss + `edge_loss` × edge_loss.
  - closure_loss: với mỗi object GT trong cửa sổ có ≥ 2 nét đúng (`oinst`), soft-min = −log Σ exp(−z) ≤ logit nét yếu nhất; BCE về 1.
    Gradient dồn vào nét object đang thiếu, nét đã bắt gần như không bị đẩy (test `test_closure_loss_sits_on_the_weakest_stroke_of_an_object`).
  - edge_loss: BCE, đích 1 khi hai nét cùng đúng và cùng `oinst`; cạnh không có nét đúng trọng số 0,1.
- Nhãn `oinst` / lớp chỉ dùng làm đích loss, không vào input. Suy luận không đổi (đồ thị nối dựng từ nét của chính trang, không cần GT).
  Logit "cùng object" chưa dùng trong hậu kỳ (để version sau: tách / gộp khối theo cạnh).

**Nguồn SOTA (bài gốc, `research/SOTA_VECTOR.md`):**
- GAT-CADNet (Zheng et al., CVPR 2022): đồ thị primitive, attention có mã hoá quan hệ không gian của cạnh, dự đoán cạnh nối hai primitive cùng
  instance (panoptic qua ma trận kề); trong bài đồ thị đưa PQ 0,602 → 0,737. Lấy: attention có bias theo cạnh + đầu cạnh cùng instance.
  Bỏ: đồ thị theo khoảng cách (dùng cạnh nối đầu mút / khe / chữ T, F22: 96–99 % cạnh nối nét cùng nhãn), cascade tinh chỉnh cạnh.
- CADTransformer (Fan et al., CVPR 2022) / VecFormer / SymPoint-V2: attention lân cận / cửa sổ trên primitive; thân U-Net cửa sổ của OpenStroke2
  đã có phần này nên không thêm.
- Loss closure: không có bài tương ứng cho CAD; cùng họ với loss MIL / soft-min "đủ cả tập" (nét yếu nhất của túi). Không lấy PQ FloorPlanCAD
  làm kỳ vọng cho F2 HVAC.
- Không chép code hay weights ngoài; tự viết (license repo).

## 5. Cách triển khai

- `openstroke/models/openstroke2.py`: `GraphAttention`, `EdgeHead`, `edge_geometry`, `flat_links`, `closure_loss`, `edge_loss`; tham số
  `rel_attn`, `edge_head` trong config / checkpoint.
- `openstroke/data.py`: `Page.oinst`, cửa sổ và batch mang `oinst` (−1 cho nét không có object / padding).
- `openstroke/train.py` / `engine/trainer.py`: cờ `rel_attn=`, `edge_head=`, `closure=`, `edge_loss=` (mặc định 0,5); khối mới thêm vào
  checkpoint cũ khởi tạo 0 (model lúc đầu đúng bằng checkpoint, test `test_graph_attention_and_edge_head_start_as_the_model_without_them`);
  đồ thị nối tự dựng và cache khi bật `rel_attn` / `edge_head` (train, `Predictor`, `openstroke scores`). `results.csv` ghi thêm cột loss mới.
- `tools/make_e10.py`: dataset e10.

## 6. Dữ liệu, split, metric

- e10 = e9g (e8 + 6 lớp chung) + cung cong khép object: nét cong nhãn `other`, hai đầu cách nét đúng ≤ 1 pt, ≥ 80 % chiều dài cách viền một
  Elbow / Elbow vuông / Tee của GT ≤ 3 pt. Bản đầu (mọi object, ≥ 50 %) thêm cả vòng đầu ống mềm ở cuối Duct và ký hiệu nhỏ (xem ảnh), nên
  siết lại. Kết quả: +152 nét cong trên trang GT (+613 pt, 0,11 % chiều dài nét đúng), +4.995 trên trang Viewer (+7.982 pt, 0,13 %). `ocls / oinst / octr` tính lại bằng `assign_objects`. Cùng nét, cùng `folds.json` (hvac10b) với e8 / e9 / e9g.
  Nguồn object: score pack 203 trang (GT AI_Takeoff chỉ đọc cho trang doc, Viewer "Labels V1" cho trang vw), như e9.
- Không augment thêm. Val 13 trang (5 GT + 8 Viewer) dùng chọn best; không có test riêng.
- Metric chính: F2 object, rule_nl 6 lớp, conf 0,5, chấm ở máy từ `val_scores.npz` của mỗi run, **hai hậu kỳ**: đầy đủ (`--post --items --fuse 0.5`
  + base logic) và chỉ khối khép kín (`--base <rỗng> --no-must`). F2 nét giữa e9g và e10 không so được (nhãn val đổi); F2 object thì so được.
- Chẩn đoán phụ: tỉ lệ object GT bắt đủ nét / một phần (probe F165), R nét cong trên viền object.

## 7. Baseline, ablation và recipe

Mọi run: khởi tạo `e114_strokes_nolayer/fold0/best.pt`, batch 8, lr0 0,001, warmup 1, 10 epoch, phần công thức còn lại từ checkpoint (như E116 / E117).

| Run | Data | Thêm | So với |
|---|---|---|---|
| E117c | e9g | – | (đối chứng, §13 OPENSTROKE29) |
| E118 | e10 | – | E117c: tác dụng nhãn cung (H134) |
| E119 | e10 | `closure=0.5` | E118: tác dụng loss closure (H133) |
| E120 | e10 | `closure=0.5 rel_attn=2 edge_head=true` | E119: tác dụng attention đồ thị + đầu cạnh (H132) |

## 8. Kỳ vọng ghi trước và tiêu chí

Kỳ vọng (ước lượng của agent, chưa đo; trần ở §2):
- E118: R nét cong trên viền object (val) tăng từ ~3–15 % lên ≥ 50 %; F2 object hậu kỳ khối kín +0,005 đến +0,02.
- E119: object GT bắt đủ nét (GT5) 75,7 % → ≥ 82 %; F2 object hậu kỳ khối kín +0,01 đến +0,02 so với E118.
- E120: F2 object hậu kỳ khối kín +0,01 đến +0,03 so với E119, chủ yếu ở P (bớt khối kín giả) và Viewer.

Tiêu chí (hậu kỳ chỉ khối khép kín, val GT5 và Viewer 8):
- H134 đạt nếu E118 ≥ E117c + 0,01 trên Viewer và không tụt quá 0,01 trên GT5; bác bỏ nếu cả hai < +0,005.
- H133 đạt nếu E119 ≥ E118 + 0,01 trên cả hai nhóm và tỉ lệ object bắt đủ nét tăng ≥ 5 điểm; bác bỏ nếu < +0,005 hoặc R nét tụt.
- H132 đạt nếu E120 ≥ E119 + 0,02 trên ít nhất một nhóm và không nhóm nào tụt quá 0,01; bác bỏ nếu cả hai < +0,01.
- Mọi run: F2 nét không tụt quá 0,01 so với run đối chứng cùng data (nguyên tắc R trước).

## 9. Kiểm thử và chi phí

`tests/test_os30.py` (7 test, CPU): khởi tạo đúng bằng checkpoint cũ; attention chỉ nhận tin từ nét được nối; padding không đổi kết quả;
closure dồn gradient vào nét thiếu; đích cạnh đúng; batch mang `oinst`; train thật tí hon (loss mới tính, log, checkpoint giữ cờ, train tiếp
từ checkpoint đó). Chi phí GPU chưa đo: 2 khối attention ở tầng 0 trên số cạnh ~ vài lần số nét, dự kiến thêm < 15 % thời gian / VRAM mỗi epoch
(E116: 3:33 / epoch, 72 GB ở batch 8); ghi số thật từ log.

## 10. Lệnh train Colab

Upload `F:\Source_code\OpenStroke\datasets\openstroke_e10_data.zip` lên `MyDrive/Dataset/` (cùng cấu trúc e9g). E117c xem §13 OPENSTROKE29.

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/os30 && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_e10_data.zip -d /content/os30
```

E118 (chỉ đổi nhãn):

```python
!openstroke train \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    data=/content/os30/e10 \
    batch=8 \
    lr0=0.001 \
    warmup_epochs=1 \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke30 \
    name=e118_curve_labels
```

E119 (+ loss closure):

```python
!openstroke train \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    data=/content/os30/e10 \
    closure=0.5 \
    batch=8 \
    lr0=0.001 \
    warmup_epochs=1 \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke30 \
    name=e119_closure
```

E120 (+ attention đồ thị + đầu cùng object):

```python
!openstroke train \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    data=/content/os30/e10 \
    closure=0.5 \
    rel_attn=2 \
    edge_head=true \
    batch=8 \
    lr0=0.001 \
    warmup_epochs=1 \
    epochs=10 \
    project=/content/drive/MyDrive/Train/OpenStroke30 \
    name=e120_graph_attn
```

Lần đầu bật `rel_attn` / `edge_head`, đồ thị nối được dựng cho từng trang và cache cạnh dataset (thêm vài phút một lần). Gửi lại mỗi run:
log và `fold0/val_scores.npz`.

## 11. Kết quả thật

Chưa có (chờ log).

## 12. Điểm yếu và hướng sau

- Hậu kỳ chỉ khối khép kín sập trên ống vẽ nét đứt (`doc000196`): đã sửa ở F166 (hợp nhất nét vẽ chồng trước khi tìm đầu hở, đầu hở từ
  1 pt, khép góc). Trang train 0,734 → 0,746, `doc000196` 0,432 → 0,757; còn kém hậu kỳ đầy đủ (0,796) ở `doc000190`, `doc000195`, `doc000196`.
- Logit "cùng object" chưa dùng trong hậu kỳ; nếu H132 đạt, dùng nó để tách / gộp khối (ống dài bị cắt, phụ kiện dính ống).
- Nhãn Viewer vẫn khác GT về Chân rẽ (12,9 % so với 4,4 % chiều dài); e10 không sửa.
