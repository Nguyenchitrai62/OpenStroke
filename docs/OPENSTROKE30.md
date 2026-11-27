# OpenStroke30: attention trên đồ thị nối đầu mút + đầu "cùng đối tượng" + loss closure + nhãn cung cong

Sổ cái: F164–F165 (thiết kế, trần), F166 (hậu kỳ nét đứt), F169–F174 (kết quả E120, E121). Giả thuyết: H132 (attention + đầu cạnh),
H133 (closure), H134 (nhãn cung), H135 (E121). Code: `openstroke/models/openstroke2.py` (`GraphAttention`, `EdgeHead`, `edge_geometry`,
`closure_loss`, `edge_loss`), `openstroke/graph.py` (đồ thị nối), `openstroke/data.py` (`oinst`), `tools/make_e10.py`, `tools/make_e11.py`.
Test: `tests/test_os30.py`. Model: `model=openstroke30`.

Ký hiệu như [OPENSTROKE46](OPENSTROKE46.md); số trong thẻ này tính trên nhãn `e10` / `e11` cũ, không so trực tiếp với v41 trở đi (`e13_*`).

## 1. Bằng chứng

Điểm `last.pt` E116 (v29), val 13 trang, TTA ×1.

| Đo | Số | Nguồn |
|---|---|---|
| Nét cong nhãn đúng: R | 0,995 (GT) / 0,996 (Viewer): model không bỏ cung, lỗi là **nhãn** | F165 |
| Cung sát viền Elbow / Tee / Chân rẽ GT (≤ 3 pt) mà nhãn `other` | GT 1.625 pt, Viewer 3.495 pt (polygon GT vẽ cung bằng vài dây cung) | F165 |
| Object GT chỉ bắt được một phần | 21 % (GT5); 48 % chiều dài nét sót có đầu mút chạm nét đúng đã bắt | F165 |
| Trần F2 object khi nét hoàn hảo, hậu kỳ khối khép kín | GT5 0,808 → 0,854, VW8 0,529 → 0,640 | F165 |
| Khối quan hệ cũ (`relations=2`, v3; `GraphMixer`, v4) | hoà trong nhiễu / chậm và sập: trung bình tin nhắn, loss vẫn chấm từng nét | F22, E14 |

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v30 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Một cạnh của Elbow nên kéo cạnh còn lại | **`GraphAttention` × 2** (`rel_attn=2`) trên đồ thị nối (đầu mút chạm, khe nét đứt, chữ T): α = softmax(q·k/√d + bias(loại cạnh, hình học 4 chiều)), chiếu ra và FFN cuối khởi tạo 0 | F2 object (khối kín) ≥ E117 + 0,02 ở một nhóm | cả hai nhóm < +0,01 |
| Cần biết hai nét có cùng đối tượng | **`EdgeHead`**: MLP trên `[h_i + h_j, |h_i − h_j|, cạnh]` → logit; BCE, đích 1 khi hai nét đúng cùng `oinst`, cạnh nền trọng số 0,1 | — | — |
| Không có gì buộc bắt đủ đối tượng | **closure** 0,5: mỗi object GT ≥ 2 nét, soft-min logit (≈ nét yếu nhất) → BCE về 1; gradient dồn vào nét đang thiếu | object bắt đủ nét GT5 75,7 % → ≥ 82 % | — |
| Nhãn bỏ cung khép object | **`e10`**: cung `other` có hai đầu cách nét đúng ≤ 1 pt và ≥ 80 % chiều dài sát viền Elbow / Elbow vuông / Tee ≤ 3 pt → đúng (+152 nét GT, +4.995 nét Viewer) | R cung trên viền ≥ 50 % | — |

`oinst` chỉ làm đích loss, không vào input. Suy luận dựng đồ thị từ nét của trang, không cần GT.

## 3. Chi tiết

- v28 + 2 `GraphAttention` (2 × 34K) + `EdgeHead` (14K). Tổng 17,21M. Khối mới khởi tạo 0: bước 0 đúng bằng checkpoint nguồn (test).
- Chi phí đo được: 4:15 / epoch, 82,4 GB ở batch 8 (E117: 3:34, 72,3 GB).

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Từ `best.pt` E114, batch 8, `lr0=0.001`, warmup 1, 10 epoch, `e10`. Theo yêu cầu, chạy E120 (gộp nhãn + closure + attention + đầu cạnh) trước,
so với E117 (cùng split, công thức, TTA ×8).

| | E117 | **Kỳ vọng E120** |
|---|---|---|
| F2 nét | 0,842 | không tụt quá 0,01 |
| F2 object, hậu kỳ khối khép kín, GT5 / VW8 | 0,804 / 0,539 | + 0,01 đến + 0,03, ≥ + 0,02 ở một nhóm |

Đạt: ≥ E117 + 0,02 ở một nhóm, không nhóm nào tụt quá 0,01. Bác bỏ cả gói: cả hai nhóm < + 0,01. Gói đạt thì chạy E118 (chỉ nhãn), E119
(+ closure) để tách phần.

## 5. Lệnh

```bash
openstroke train data=/content/e10 model=<best.pt của E114> closure=0.5 rel_attn=2 edge_head=true batch=8 lr0=0.001 \
    warmup_epochs=1 epochs=10 name=e120_graph_attn
```

## 6. Kết quả thật

**E120 (F169–F172):** 10 epoch, 0,81 h. `best.pt` = epoch 6.

| | Kỳ vọng | E120 |
|---|---|---|
| F2 nét, TTA ×8 | ≥ 0,832 | 0,857 (R 0,907, P 0,703) |
| F2 object, khối khép kín, GT5 / VW8 | ≥ + 0,02 ở một nhóm | 0,815 / 0,546 (+ 0,011 / + 0,007) |
| Object GT bắt đủ nét (GT5) | ≥ 82 % | 84,0 % (E117 77,2 %) |

- Vùng xám: không đạt (+ 0,02) và không bị bác bỏ (cả hai nhóm < + 0,01 mới bác bỏ). Chưa chạy E118 / E119 nên chưa tách được phần của
  từng thành phần.
- Ngưỡng tốt nhất dịch lên 0,65–0,85 (E117 0,05–0,20): closure đẩy xác suất nét của object lên.
- F2 nét + 0,015 nhưng F2 object chỉ + 0,01: nét bắt thêm (đường nối ống, nét ngang trong ống) cắt một ống thành nhiều khối,
  Duct dự đoán 749 → 847 (GT 644).

**E121 (F173–F174, H135):** cùng công thức trên `e11` (thêm vách ngoài + đường chéo của Elbow vuông vẽ bằng một item: +38 item GT, +7 Viewer).
F2 nét 0,853 (E120 0,857). F2 object hậu kỳ mặc định GT5 0,820 / VW8 0,556 (E120 0,827 / 0,544). Mọi item đổi nhãn đều ở trang train nên
tiêu chí chính (R item trên val) không đo được; không bị bác bỏ. Giữ `e11` làm nhãn gốc.
