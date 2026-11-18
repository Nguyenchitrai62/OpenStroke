# OpenStroke32 — tổng quát hóa: trộn trang train (đổi vùng / đặt object GT lên nền trang khác)

Ngày: 2026-11-18. Trạng thái: đã code và test CPU, chưa train. Sổ cái F182–F183, giả thuyết H137 (E123) và H138 (E124, ý của user).

## 1. Mục tiêu và phạm vi

Mục tiêu chung không đổi: F2 object ≥ 0,80 trên GT giữ ngoài train, và chịu được bản vẽ lạ. Version này nhắm **khoảng cách giữa trang đã
học và trang lạ** của model nét. Mạng giữ nguyên openstroke31 (E122); chỉ thêm một augment khi train. Không đổi dữ liệu, split, nhãn, cách chấm.

## 2. Bằng chứng và điểm yếu version trước

**Model thuộc bài trang train (F182, điểm E120 đã lưu, ngưỡng 0,70, nhãn e10):**

| Trang | F2 nét | R | P | Tỉ lệ chiều dài nét chọn có p ≥ 0,95 |
|---|---|---|---|---|
| 19 trang GT train | 0,993 | 0,997 | 0,985 | 99 % |
| 5 trang GT val (PDF chưa thấy) | 0,933 | 0,968 | 0,844 | 78 % |
| 8 trang Viewer val (log E122) | 0,63–0,86 | | | |

**Chỗ lệch không nằm ở kiểu nét (F183, điểm E122 val, ngưỡng 0,75, nhãn e11):**
- Trang GT val: R ≥ 0,95 ở mọi nhóm độ dày, kể cả nét mảnh (34 % chiều dài đúng ở val so với 17 % ở train). Lỗi là P: nét cùng kiểu với ống
  (dày 0,5–1,5, dài 15–80 pt) bị bắt thừa, P 0,70–0,79.
- Trang Viewer val: bỏ sót nét thẳng 15–80 pt (R 0,61), bắt thừa nét màu (P 0,33) và nét mảnh (P 0,32).
- Model nhận ra kiểu nét, nhưng chưa phân biệt được ống với vật thể khác vẽ cùng kiểu trên bản vẽ lạ. Nó học thuộc cả nền của từng trang.
  Augment kiểu nét (restyle đã có, 0,5) không nhắm đúng chỗ này.

**Các lần trước:** tiền huấn luyện không nhãn là cải thiện tổng quát hóa rõ nhất của repo (E19 / E26, +0,024–0,03), nhưng thêm nguồn dữ liệu nên
chờ user (AGENTS 0.2). Copy-paste object là tín hiệu tăng đầu tiên của nhánh detector (E106, +0,03 rule_nl, F117), chưa từng thử cho model nét.

## 3. Giả thuyết đăng ký trước

**H137 (E123, augment):** khi một phần cửa sổ train được thay bằng vùng của trang train khác lấy quanh một object GT, model gặp object và nền
của các bản vẽ khác nhau cạnh nhau, không còn dựa được vào việc thuộc một trang. P trên trang GT lạ tăng, R không giảm. Biến duy nhất so
với E122: `paste=0.5`.

**H138 (E124, augment, user 2026-11-18):** tách hẳn object khỏi nền: cửa sổ chỉ giữ nét other của trang này, cộng toàn bộ nét true (object GT)
của một trang train khác, xoay / lật / dời ngẫu nhiên. Model không còn nền nào quen để dựa vào, phải nhận ống từ chính nét ống và quan hệ giữa
chúng, nên P trên trang lạ tăng mạnh hơn E123. Biến duy nhất so với E122: `paste=0.5 paste_mode=fg`.

## 4. Kiến trúc và luồng dữ liệu

Mạng openstroke31 giữ nguyên. Augment mới ở bước cắt cửa sổ (`data.WindowDataset.__getitem__` → `data.region_swap`), trước các augment khác:

```
cửa sổ host (trang p, 65.536 nét)          cửa sổ donor (trang q ≠ p, chọn theo cùng xác suất lấy trang)
   │ chọn hình vuông cạnh s = U(0,2; 0,5) × chiều dài lớn nhất của cửa sổ, vị trí ngẫu nhiên
   │ bỏ mọi nét có tâm trong hình vuông             hình vuông cùng cỡ, tâm ở một nét ĐÚNG ngẫu nhiên của donor
   └──────────────── ghép ◄──────────────── tịnh tiến các nét donor trong hình vuông vào chỗ trống
→ một cửa sổ: nhãn, lớp / tâm object (dời theo), chỉ số object (cộng thêm để không trùng host), cạnh nối giữa các nét donor còn lại,
  nhóm kiểu nét; vượt 65.536 nét thì bỏ ngẫu nhiên nét host
→ augment cũ (lật, xoay 90°, co giãn, màu, restyle, bỏ nét) → model (nhánh ảnh vẽ lại từ chính các vector của cửa sổ)
```

- Ảnh của model vẽ từ vector của cửa sổ, nên ảnh, vector và nhãn đổi cùng nhau (AGENTS 0.4). Không dùng nhãn val, không đổi GT gốc.
- Nét giữ đặc trưng trang / lân cận của trang gốc (đặc trưng lân cận ở mép vùng ghép không khớp hoàn toàn). Đây là nhiễu có chủ ý nhỏ;
  nếu H137 đạt mới cân nhắc tính lại đặc trưng lân cận sau khi ghép.
- Số nét gần như không đổi (thay chứ không thêm), nên VRAM không tăng (E122: 82,4 / 96 GB).

**Chế độ `fg` (E124, `data.fg_swap`):** cửa sổ host giữ mọi nét other, bỏ mọi nét true. Mọi nét true của cửa sổ donor (object GT, cạnh nối
giữa chúng, lớp / tâm / chỉ số object) được xoay k × 90°, lật ngẫu nhiên, rồi dời để tâm của cụm rơi vào tâm cửa sổ host ± 25 % kích thước
(`FG_SHIFT`). Vượt 65.536 nét thì bỏ ngẫu nhiên nét host (rồi nét donor nếu vẫn thừa). `paste_mode=both`: mỗi cửa sổ bốc ngẫu nhiên một trong hai.
Khác `region`: object của donor không mang theo nền gốc của nó (mất các nét other gắn với ống như chữ kích thước, cách nhiệt, miệng gió), và
mật độ ở vùng chồng tăng. Đây là đánh đổi được đo bằng E124 so với E123.

**Nguồn SOTA (bài gốc):**
- Simple Copy-Paste (Ghiasi et al., CVPR 2021): dán object của ảnh này sang ảnh khác cho instance segmentation, tăng mạnh nhất khi ít dữ liệu.
  Lấy: dán object kèm nhãn instance. Thay cho vector: dán cả vùng quanh object (cả nền của donor), vì ranh giới object HVAC dùng chung nét với
  ống bên cạnh; dán riêng nét của object sẽ để lại nét cụt.
- CutMix (Yun et al., ICCV 2019): thay một hình chữ nhật của ảnh bằng vùng của ảnh khác, nhãn theo diện tích. Lấy: thay vùng để giữ mật độ.
  Bỏ: trộn nhãn theo tỉ lệ (ở đây nhãn từng nét giữ nguyên, không cần trộn).
- Không chép code ngoài.

## 5. Cách triển khai

- `openstroke/data.py`: `Augment.paste`, `PASTE_SIDE`, `ROW_KEYS`, `region_swap`; `WindowDataset.__getitem__` gọi khi `paste > 0` (cả đường
  dựng batch trên CPU và trên GPU, vì đổi vùng làm trước khi trả cửa sổ thô).
- `openstroke/data.py`: `fg_swap`, `_turn_rows`, `FG_SHIFT`; `Augment.paste_mode` (`region` | `fg` | `both`).
- `openstroke/engine/trainer.py`, `openstroke/train.py`: cờ `paste=` (mặc định 0) và `paste_mode=` (mặc định `region`), log `page mix <p> (<mode>)`,
  kế thừa khi train từ checkpoint.
- Không thêm tham số mạng; checkpoint vẫn là openstroke31.

## 6. Dữ liệu, split, augmentation và metric

- Dữ liệu e11, split hvac10b như E122: train 19 GT + 171 Viewer, val 5 GT + 8 Viewer. Donor chỉ lấy từ trang train.
- Metric chính (như E122):
  - F2 nét val (TTA ×8, log train);
  - F2 object rule_nl 6 lớp, conf 0,5, hậu kỳ mặc định của app (logic trên nét của model + khối kín, không bắt buộc phủ), nhóm GT5 / Viewer 8,
    chấm ở máy từ `val_scores.npz`;
  - P / R nét trên trang GT val (chỗ lệch chính của F183).
- Phụ: khoảng cách train / val. Cần `openstroke scores` trên 24 trang GT cho E122, E123 và E124 (chạy trên GPU, lệnh ở §10).

## 7. Baseline, ablation và recipe

| Run | Khởi tạo | Data | Cờ | Epoch |
|---|---|---|---|---|
| E122 (đối chứng, đã có) | E114 | e11 | `closure=0.5 rel_attn=2 edge_head=true edge_pairs=true` | 15 |
| **E123** | E114 | e11 | E122 + `paste=0.5` | 15 |
| **E124** | E114 | e11 | E122 + `paste=0.5 paste_mode=fg` | 15 |

E123 và E124 chạy song song, cùng xác suất trộn để so được hai kiểu trộn với nhau. Nếu cả hai đạt, run sau dùng `paste_mode=both` với xác suất cao hơn.

Batch 8, lr0 0,001, warmup 1 như E122. Cùng 15 epoch để lịch LR giống đối chứng.

## 8. Kỳ vọng và tiêu chí đạt / bác bỏ

Kỳ vọng (ước lượng của agent, chưa đo): F2 nét +0,005 đến +0,02; P nét GT val +0,02 đến +0,05; F2 object GT5 +0,005 đến +0,02; Viewer không đổi
hoặc tăng nhẹ. F2 nét theo epoch có thể thấp hơn E122 ở vài epoch đầu (cửa sổ khó hơn).

Tiêu chí (ghi trước khi train):
- **Đạt:** F2 nét val ≥ 0,860 (E122 0,850), **hoặc** P nét GT val tăng ≥ 0,03 với R nét GT val không giảm quá 0,01; **và** F2 object rule_nl
  GT5 ≥ 0,845 (E122 0,835) với Viewer không tụt quá 0,01.
- **Bác bỏ:** F2 nét val < 0,845, hoặc F2 object tăng < 0,005 ở cả GT5 và Viewer.
- Ở giữa: vùng xám, ghi số; nếu khoảng cách train / val (khi có điểm 24 trang) thu hẹp ≥ 0,02 thì vẫn giữ augment cho các run sau.
- **H138 (E124):** cùng mốc đạt / bác bỏ như H137 so với E122. Thêm: E124 so với E123, chênh F2 object < 0,01 ở cả hai nhóm coi là ngang nhau.
  F2 nét theo epoch của E124 có thể thấp hơn E123 ở các epoch đầu (cửa sổ khác xa trang thật hơn).

## 9. Kiểm thử và chi phí

`tests/test_os32.py` (6 test, CPU): đổi vùng thay đúng hình vuông bằng vùng donor quanh nét đúng, dời tâm object theo nét, chỉ số object donor
không trùng host, cạnh nối chỉ giữa nét còn trong cửa sổ; giữ trần số nét và bỏ qua donor không có nét đúng; `WindowDataset` đổi vùng, tái lập
theo seed, ghép batch được; train tí hon qua trainer thật với `paste=1` (log và `args.json` giữ cờ); `fg_swap` chỉ giữ nét other của host và nét
true của donor, giữ khoảng cách giữa các nét donor (dời cứng), cạnh nối giữa object donor, trần số nét; chế độ `fg` / `both` trong `WindowDataset`. Toàn bộ test: xem nhật ký phát hành.

Chi phí: thêm một lần cắt cửa sổ donor cho một nửa số mẫu trên CPU worker; mạng và VRAM không đổi. Dự kiến cùng thời gian epoch với E122
(4:15) nếu worker không nghẽn; ghi số thật từ log.

## 10. Lệnh train Colab

Dữ liệu như E122: `MyDrive/Dataset/openstroke_e11_data.zip`.

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/os30 && unzip -q -o /content/drive/MyDrive/Dataset/openstroke_e11_data.zip -d /content/os30
```

```python
!openstroke train \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    data=/content/os30/e11 \
    closure=0.5 \
    rel_attn=2 \
    edge_head=true \
    edge_pairs=true \
    paste=0.5 \
    batch=8 \
    lr0=0.001 \
    warmup_epochs=1 \
    epochs=15 \
    project=/content/drive/MyDrive/Train/OpenStroke32 \
    name=e123_region_swap
```

E124 (chạy song song với E123, chỉ thêm `paste_mode=fg`):

```python
!openstroke train \
    model=/content/drive/MyDrive/Train/OpenStroke28/e114_strokes_nolayer/fold0/best.pt \
    data=/content/os30/e11 \
    closure=0.5 \
    rel_attn=2 \
    edge_head=true \
    edge_pairs=true \
    paste=0.5 \
    paste_mode=fg \
    batch=8 \
    lr0=0.001 \
    warmup_epochs=1 \
    epochs=15 \
    project=/content/drive/MyDrive/Train/OpenStroke32 \
    name=e124_fg_swap
```

Sau khi train xong, điểm trên 24 trang GT (train + val) của E123, E124 và đối chứng E122, để đo khoảng cách train / val và chọn hậu kỳ trên
trang train:

```python
!openstroke scores \
    model=/content/drive/MyDrive/Train/OpenStroke32/e123_region_swap/fold0/best.pt \
    data=/content/os30/e11
```

```python
!openstroke scores \
    model=/content/drive/MyDrive/Train/OpenStroke32/e124_fg_swap/fold0/best.pt \
    data=/content/os30/e11
```

```python
!openstroke scores \
    model=/content/drive/MyDrive/Train/OpenStroke31/e122_edge_pairs/fold0/best.pt \
    data=/content/os30/e11
```

Gửi lại: log train và `fold0/val_scores.npz` của E123 và E124, và ba file `all_scores.npz` (mỗi file nằm cạnh `best.pt` của run đó).

## 11. Kết quả thật từ log

**E123** (log user 2026-11-18, run `e123_region_swap3`). Lệch thẻ: **batch 4** thay vì 8 (E122 batch 8), nên mỗi epoch chỉ thấy nửa số cửa sổ;
2:15 / epoch, 46,1 GB, 0,68 h.

- F2 nét val mỗi epoch (chưa TTA): 0,834 / 0,843 / 0,840 / 0,852 / **0,853** / 0,851 … 0,848; phẳng từ epoch 4. `best.pt` = epoch 5, F2 nét + TTA ×8
  **0,854** (R 0,913, P 0,680, ngưỡng 0,70); E122 0,850 (R 0,891, P 0,717).
- Theo trang so với E122:

| Trang | E122 F2 (R / P) | E123 F2 (R / P) |
|---|---|---|
| doc180 | 0,895 (0,977 / 0,671) | 0,889 (0,969 / 0,668) |
| doc187 | 0,938 (0,949 / 0,897) | 0,956 (0,977 / 0,881) |
| doc189 | 0,901 (0,949 / 0,750) | 0,905 (0,966 / 0,722) |
| doc200 | 0,945 (0,970 / 0,856) | 0,932 (0,968 / 0,808) |
| doc201 | 0,957 (0,971 / 0,905) | 0,952 (0,965 / 0,901) |
| vw_2f3d p4 | 0,719 (0,707 / 0,770) | 0,742 (0,741 / 0,749) |
| vw_579f p2 | 0,666 (0,651 / 0,733) | 0,708 (0,703 / 0,728) |
| vw_62b1 p2 | 0,728 (0,963 / 0,368) | 0,710 (0,963 / 0,346) |
| vw_62b1 p3 | 0,627 (0,983 / 0,256) | 0,622 (0,984 / 0,251) |
| vw_e241 p1 | 0,724 (0,696 / 0,864) | 0,806 (0,797 / 0,843) |
| vw_e241 p2 | 0,688 (0,679 / 0,729) | 0,708 (0,713 / 0,687) |
| vw_e3fe p2 | 0,863 (0,950 / 0,630) | 0,858 (0,959 / 0,604) |
| vw_f4af p4 | 0,847 (0,952 / 0,588) | 0,812 (0,973 / 0,490) |

- Đọc: phần tăng nằm ở **R trên trang Viewer** mà E122 bỏ sót (vw_e241 p1 +0,10, vw_579f +0,05, vw_2f3d +0,03). **P trên trang GT không tăng**
  (doc200 −0,05, doc189 −0,03): ngược với điều H137 dự đoán (P GT tăng).
- Tiêu chí mức nét của H137: F2 nét 0,854 < 0,860 và P GT không tăng ≥ 0,03, nên **chưa đạt ở mức nét**; F2 nét ≥ 0,845 nên chưa bị bác bỏ.
  Kết luận chờ F2 object rule_nl từ `val_scores.npz`. So sánh lẫn batch 4 / 8: E123 thấy nửa số cửa sổ của E122 mà F2 nét ngang.
- **F2 object rule_nl** (13 trang val, 6 lớp, conf 0,5, nét ≥ 0,70; logic chạy lại trên nét E123; `runs_colab/e123/*.txt`), GT5 / Viewer 8:

| Hậu kỳ | E122 | E123 |
|---|---|---|
| Mặc định app (logic + khối kín, không bắt buộc phủ) | 0,835 / 0,532 | 0,833 / **0,550** |
| Chỉ khối khép kín | 0,825 / 0,533 | 0,827 / **0,551** |

- **H137: vùng xám.** Không đạt (F2 nét < 0,860, P GT không tăng, F2 object GT5 < 0,845); không bác bỏ (Viewer +0,018 ≥ 0,005). Augment giúp trang
  Viewer (R nét +0,02, F2 object +0,018), không đổi trang GT. Khoảng cách train / val chưa đo (chưa có `all_scores.npz`). Lệch batch 4 / 8 vẫn còn.

**E124** (`paste_mode=fg`, log user 2026-11-18, run `e124_fg_swap2`, batch 4 như E123; 2:17 / epoch, 56,6 GB, 0,62 h).
- F2 nét val mỗi epoch: 0,791 / 0,798 / 0,815 / 0,823 / 0,839 … 0,841 (phẳng từ epoch 5, thấp hơn E123 suốt run). `best.pt` = epoch 13, + TTA ×8
  **0,844** (R 0,898, P 0,679). doc187 R 0,977 → 0,892; Viewer vw_2f3d 0,751, vw_579f 0,723.
- **BCE train thấp hơn hẳn** (epoch 15: E124 0,0235, E123 0,0376, E122 0,0384) trong khi val kém hơn: cửa sổ fg dễ hơn cửa sổ thật.
- F2 object rule_nl (cùng giao thức, logic chạy lại trên nét E124 ở 0,70): mặc định 0,824 / **0,552**, chỉ khối kín 0,813 / 0,553
  (E122 0,835 / 0,532; E123 0,833 / 0,550).
- **H138 bác bỏ** theo tiêu chí ghi trước (F2 nét 0,844 < 0,845); F2 object GT5 −0,011, Viewer +0,020.

## 12. Kết luận và version tiếp theo

- Cả hai kiểu trộn tăng F2 object trang Viewer (+0,018 / +0,020) và R nét trên đúng các trang Viewer bị bỏ sót, nhưng **không cải thiện trang GT**
  (E123 −0,002, E124 −0,011). H137 vùng xám, H138 bác bỏ.
- **Rò đặc trưng (giả thuyết, F185):** mỗi nét mang đặc trưng tính trên trang gốc của nó: tỉ lệ nhóm kiểu nét trên trang (`style_shape_share`,
  `style_len_share`), thứ tự vẽ trên trang (`seqno_rank`) và 6 đặc trưng lân cận (`nb10_log` … `nn_dist_log`). Ở chế độ fg, MỌI nét true đến từ trang
  khác, nên các đặc trưng này tách nét true khỏi nét nền mà không cần nhìn hình dạng: BCE train của E124 thấp hẳn, val kém đi. Ở chế độ region chỉ
  một mảng bị lệch nên rò ít hơn.
- **Lỗi code tìm được:** `feature_drop=` trên lệnh bị bỏ qua khi train từ checkpoint (model giữ mặt nạ của checkpoint; log vẫn in giá trị mới).
  Không ảnh hưởng E120–E124 (chúng dùng đúng `layer_stats` như checkpoint E114). Sửa ở OpenStroke33.
- Version sau: OpenStroke33 bỏ các đặc trưng phụ thuộc trang (nhóm `context`) để trộn trang không còn rò, rồi trộn lại (`paste_mode=both`).
