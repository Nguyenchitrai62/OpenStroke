# OpenStroke32 — tổng quát hóa: đổi vùng giữa các trang train (copy-paste object GT cho cửa sổ vector)

Ngày: 2026-11-18. Trạng thái: đã code và test CPU, chưa train. Sổ cái F182–F183, giả thuyết H137.

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
- `openstroke/engine/trainer.py`, `openstroke/train.py`: cờ `paste=` (mặc định 0), log `region swap <p>`, kế thừa khi train từ checkpoint.
- Không thêm tham số mạng; checkpoint vẫn là openstroke31.

## 6. Dữ liệu, split, augmentation và metric

- Dữ liệu e11, split hvac10b như E122: train 19 GT + 171 Viewer, val 5 GT + 8 Viewer. Donor chỉ lấy từ trang train.
- Metric chính (như E122):
  - F2 nét val (TTA ×8, log train);
  - F2 object rule_nl 6 lớp, conf 0,5, hậu kỳ mặc định của app (logic trên nét của model + khối kín, không bắt buộc phủ), nhóm GT5 / Viewer 8,
    chấm ở máy từ `val_scores.npz`;
  - P / R nét trên trang GT val (chỗ lệch chính của F183).
- Phụ: khoảng cách train / val. Cần `openstroke scores` trên 24 trang GT cho cả E122 và E123 (chạy trên GPU, lệnh ở §10).

## 7. Baseline, ablation và recipe

| Run | Khởi tạo | Data | Cờ | Epoch |
|---|---|---|---|---|
| E122 (đối chứng, đã có) | E114 | e11 | `closure=0.5 rel_attn=2 edge_head=true edge_pairs=true` | 15 |
| **E123** | E114 | e11 | E122 + `paste=0.5` | 15 |

Batch 8, lr0 0,001, warmup 1 như E122. Cùng 15 epoch để lịch LR giống đối chứng.

## 8. Kỳ vọng và tiêu chí đạt / bác bỏ

Kỳ vọng (ước lượng của agent, chưa đo): F2 nét +0,005 đến +0,02; P nét GT val +0,02 đến +0,05; F2 object GT5 +0,005 đến +0,02; Viewer không đổi
hoặc tăng nhẹ. F2 nét theo epoch có thể thấp hơn E122 ở vài epoch đầu (cửa sổ khó hơn).

Tiêu chí (ghi trước khi train):
- **Đạt:** F2 nét val ≥ 0,860 (E122 0,850), **hoặc** P nét GT val tăng ≥ 0,03 với R nét GT val không giảm quá 0,01; **và** F2 object rule_nl
  GT5 ≥ 0,845 (E122 0,835) với Viewer không tụt quá 0,01.
- **Bác bỏ:** F2 nét val < 0,845, hoặc F2 object tăng < 0,005 ở cả GT5 và Viewer.
- Ở giữa: vùng xám, ghi số; nếu khoảng cách train / val (khi có điểm 24 trang) thu hẹp ≥ 0,02 thì vẫn giữ augment cho các run sau.

## 9. Kiểm thử và chi phí

`tests/test_os32.py` (4 test, CPU): đổi vùng thay đúng hình vuông bằng vùng donor quanh nét đúng, dời tâm object theo nét, chỉ số object donor
không trùng host, cạnh nối chỉ giữa nét còn trong cửa sổ; giữ trần số nét và bỏ qua donor không có nét đúng; `WindowDataset` đổi vùng, tái lập
theo seed, ghép batch được; train tí hon qua trainer thật với `paste=1` (log và `args.json` giữ cờ). Toàn bộ test: xem nhật ký phát hành.

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

Sau khi train xong, điểm trên 24 trang GT (train + val) của E123 và của đối chứng E122, để đo khoảng cách train / val và chọn hậu kỳ trên trang train:

```python
!openstroke scores \
    model=/content/drive/MyDrive/Train/OpenStroke32/e123_region_swap/fold0/best.pt \
    data=/content/os30/e11
```

```python
!openstroke scores \
    model=/content/drive/MyDrive/Train/OpenStroke31/e122_edge_pairs/fold0/best.pt \
    data=/content/os30/e11
```

Gửi lại: log train, `fold0/val_scores.npz`, và hai file `all_scores.npz` (mỗi file nằm cạnh `best.pt` của run đó).

## 11. Kết quả thật từ log

Chưa train.

## 12. Kết luận và version tiếp theo

Chưa có kết quả. Nếu H137 đạt: tính lại đặc trưng lân cận sau khi ghép, thử tỉ lệ `paste` cao hơn, rồi chọn hậu kỳ trên điểm trang train của
model mới. Nếu bác bỏ: khoảng cách là do thiếu đa dạng bản vẽ thật, cần tiền huấn luyện không nhãn trên trang HVAC (chờ user về nguồn dữ liệu).
