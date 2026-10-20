# Sổ cái nghiên cứu OpenStroke (LEDGER)

**Luật (bắt buộc với mọi agent):**
1. Mỗi thí nghiệm / job xong: thêm 1 dòng vào §2, kèm số và kết luận.
2. Mỗi phát hiện mới, lỗi mới hay hướng bị bác bỏ: thêm vào §3, §4 hoặc §5 kèm bằng chứng. Không xoá dòng cũ; nếu sai thì
   ghi "VÔ HIỆU" và lý do.
3. Giả thuyết và tiêu chí ghi trước ở `research/HYPOTHESES.md`; thẻ version ở `docs/OPENSTROKE<N>.md`.
4. Chọn việc kế tiếp từ §6 (hướng mở, xếp theo bằng chứng). Cập nhật §6 sau mỗi kết quả.

Quy ước: metric = recall / precision / F2 của `true` **theo chiều dài**, micro trên mọi trang val của mọi fold. "Trang" = một
bản GT của một trang PDF (`stem` dạng `doc000035_p0001_gt83`). Fold đang là **đề xuất** (`status: proposal`) cho tới khi user chốt;
mọi số trên fold đề xuất ghi "(tạm)".

## 1. Dữ liệu

| Bộ | Nội dung | Nơi | Ghi chú |
|---|---|---|---|
| E0 `openstroke.e0.v1` | 41 bản GT / 36 trang vật lý / 26 PDF; 5.27M shape | `datasets/e0/` (gitignored, 118 MB) | `py -3.12 -m openstroke build-e0`; QA `research/E0_QA.md` |
| Gói Colab | code + E0 + `folds.json` (đề xuất) | `datasets/openstroke_e0_colab.zip` | `py -3.12 -m openstroke pack` |

## 2. Thí nghiệm

| Ngày | Mã | Thay đổi | So với | Kết quả | Kết luận |
|---|---|---|---|---|---|
| 10-20 | E0 | bộ chuyển GT + payload → nhãn theo shape | – | 41/41 bản GT đạt chốt toạ độ (identity); FIRE 8,1 % shape / 10,9 % dài true; HVAC 15,2 % / 14,0 % | dataset sẵn sàng; chờ user quyết mục 3 |
| 10-20 | E2-v1 (user, G4) | v1, fold 0 (H1_KS + 02.DN-TT-FP), batch 8 × 4096, lr 1e-3, 30 ep (log tới ep 22) | – | val F2 **0,810 @ ep 5** (R 0,920 P 0,547, t 0,15), ep 10 0,809, ep 15 0,784, ep 20 0,788; loss train 0,32 → 0,095; VRAM 5 / 96 GB | overfit sớm; chưa có 5 fold (tạm, fold đề xuất) |
| 10-20 | E2-v1 5 fold (user, G4) | v1 mặc định (A), code trước `ed0c90c` (trainer cũ), 30 ep, fold đề xuất | E1.1 trên cùng trang | best F2 theo fold (epoch): **0,810** (5) / **0,697** (20) / **0,757** (10) / **0,680** (15) / **0,599** (5, log dừng ep 29). Micro (tính từ R, P của từng fold × chiều dài true của fold): **R 0,783 P 0,529 F2 0,714**, lạc quan vì chọn cả epoch lẫn ngưỡng trên val. Trần layer cùng trang theo fold: 0,804 / 0,512 / 0,858 / 0,736 / 0,793 | v1 **dưới** trần layer (oracle) 0,018 micro; thắng rõ ở fold 1 (2 trang không layer), thua 0,06–0,19 ở fold 2–4. H3 **bác bỏ (tạm)**: recall 0,783 < 0,85. H1 / H4 / H5 chưa chấm (thiếu GBDT, hạt giống B); H2 chưa chấm (log không có bảng theo trang). F11–F14 |
| 10-20 | probe style | bảng tra: tỷ lệ true theo style trên train → điểm cho val, không học (`research/probes/style_probe.py`) | v1, E1.1 | micro F2: style đầy đủ (màu, fill, độ dày, kiểu) **0,330**; bỏ màu 0,439; bỏ màu + mode 0,472 | nhớ style không chuyển được giữa PDF; v1 (0,714) học nhiều hơn style. F12, F14 |
| 10-20 | E1.1 | trần chọn layer bằng nhãn GT (tạm: fold đề xuất, 31 trang, bỏ 5 trang trùng) | – | **R 0,875 P 0,443 F2 0,732**; doc 35 (1 layer) 0,305, doc 36 (1 layer) 0,147; tốt nhất 0,897 (Y1 tr. 9) | lọc theo layer, kể cả chọn hoàn hảo, mất nhiều precision; trang không layer không cứu được. Lưu ý: khác thước F2 pipeline của AI_Takeoff (doc 60: 0,737 ở đây, 0,97 ở pipeline) |

## 3. Phát hiện (có bằng chứng)

- **F1. Payload `split_countfire` và GT cùng hệ display top-left trên cả 41 bản GT.** Chốt toạ độ bằng mực vector (không render,
  không PyMuPDF) chọn `identity` mọi trang, độ phủ GT 90,5–100 % ở 0,5 pt, lift 3,1–37,7 so với runner-up 0,99–3,5. Kể cả trang
  `/Rotate 270` (333.pdf doc 20, H1_KS.pdf). Xác nhận đo tay của AGENTS.md mục 3.
- **F2. Dung sai 2 pt quá lỏng để chốt toạ độ trên bản vẽ dày.** Ở 2 pt, 68–75 % điểm *ngẫu nhiên* đã nằm trên một nét, trần lift
  còn khoảng 1,4 và quy tắc kết luận (lift ≥ 1,25 × runner-up) từ chối oan doc 20 (identity 98,9 % phủ, lift 1,33 / 1,28) và doc 22
  (identity 100 %, flip_y lift 1,49 > identity 1,47). Ở 0,5 pt cả hai rõ ràng (lift 2,43 / 1,56 và 3,15 / 1,46). Chốt dùng 0,5 pt;
  nhãn vẫn dùng 2 pt.
- **F3. Cách chấm của `calibrate_rotation` (AI_Takeoff) bỏ qua điểm bị đẩy ra ngoài trang.** Trên doc 41 (2384 × 1684),
  `rot90ccw` chỉ giữ 900 / 4400 điểm GT trên trang, phủ 75 % số đó trong vùng mực thưa nên lift 5,84 > identity 3,66 (phủ 100 %
  của 4400 điểm), và bản đầu của bộ chuyển đã xoay GT sai (phủ GT còn 24,6 %, 100 / 163 object trống). Sửa: điểm ngoài trang tính
  là trượt → identity. Cùng điểm yếu tồn tại trong `AI_Takeoff/BE/app/geometry.py` (không sửa ở đây; ghi cho AI_Takeoff).
- **F4. Tỷ lệ true lệch mạnh theo trang → GT có thể không phủ đủ trang.** 10 trang `Y1_10-20.pdf` (HVAC, 16–56 object mỗi
  trang), `05. CHI TIET CAN HO` và `26 M-MH-2007.1` có 1,3–3,6 % chiều dài true, dưới 1/3 trung vị HVAC (15 %). `TOWER Z35.1`
  (FIRE) 3,3 %. Nếu GT chỉ vẽ một phần, quy tắc "còn lại là other" dạy sai (AGENTS.md mục 3.2): cần user xác nhận.
- **F5. Nét `other` cùng style với ống nhưng cách GT > 25 pt:** lớn nhất ở `Y1` trang 6 (25,5 % chiều dài true, layer `INNO-HÚT
  KHÓI`), `05. CHI TIET` (21,3 %), `Y1` trang 3 (13,4 %), `7. FP_page 16` (12,2 %, style fill đen w = 0, 850 shape), doc 41
  (11,1 %). Vùng toạ độ trong `research/E0_QA.md` §3 để user mở trên UI GT.
- **F6. Năm cặp GT trùng khác nhau đáng kể:** IoU chiều dài true 0,857–0,941 (NOBU [24] thấp nhất: bản doc 25 có thêm 9,6k pt
  true); MB TH4 bản doc 56 là tập cha của doc 23 (0 pt chỉ-A). 333.pdf: 349 so với 132 `Line` (số IoU trong `E0_QA.md` §4).
- **F7. Không trang nào có GT trống** (object GT có < 20 % viền nằm trên nét true): GT luôn được vẽ trên nét có thật.

- **F8. Cửa sổ 4096 shape của v1 nhìn không hết một mạng ống.** `research/probes/context_probe.py`: cửa sổ z-order 4096 / 16384 / 65536
  shape phủ đường chéo trung vị 507 / 1438 / 2433 pt (trang 2739 pt). Mạng ống liền (nét true nối trong 2 pt) có đường chéo trung vị theo
  chiều dài 856 pt. Phần chiều dài true thuộc mạng dài hơn cửa sổ: trung vị **67 % / 26 % / 0 %**; FIRE 94–100 % ở 4096 và 16384.
  Y1 (HVAC) 0 % ở mọi cửa sổ (mạng 150–250 pt). Ngoại lệ ở 65536: H1_KS 97 %, HVAC-B2 46 %.
- **F9. Pooling lưới theo dịch mã z-order trên trang thật** (cửa sổ 65536): lưới (1, 2, 8, 32, 128) pt cho 65536 → 6k–31k → 1,6k–5,5k →
  270–810 → 31–99 token (nén 3–10× mỗi tầng); lưới (1, 8, 32, 128) nén 15–40× ngay tầng đầu. Dữ liệu ngẫu nhiên đều cho số khác hẳn
  (65536 → 46706 ở 8 pt): không ước lượng pooling bằng dữ liệu giả.
- **F10. v1 overfit sớm** (dòng E2-v1): best ở epoch 5 / 22, loss train tiếp tục giảm 45 % mà val F2 giảm 0,02–0,03. 29 trang, không
  augment màu / tỉ lệ, không EMA, không drop path.

- **F11. v1 thắng cascade ở đúng chỗ cascade sập, thua ở trang có layer tốt.** Fold 1 (doc 35 / 36 chỉ 1 layer, `05. CHI TIET`,
  `C3-CH01`): v1 0,697, trần layer 0,512. Fold 2 (10 trang Y1 + HVAC-B2 + BLOCK A): 0,757 so với 0,858; fold 4 (7 PDF của 7 dự án
  khác nhau): 0,599 so với 0,793. Trần layer là oracle dùng nhãn GT, cascade thật thấp hơn; nhưng một model tốt phải đạt gần trần
  này trên trang có layer rõ.
- **F12. Style của ống thay đổi theo PDF; fold 3 / 4 là fold lệch style.** Phần chiều dài true của val có style đầy đủ (màu nét,
  fill, độ dày, kiểu) **chưa từng gặp** trong train: fold 0–2 1,2–4,0 %, fold 3 **51,4 %**, fold 4 **32,8 %**; theo trang: doc 60
  (0875-FP) 97,8 %, doc 22 72,0 %, doc 47 67,5 %, doc 66 65,3 %, doc 30 53,9 %. Đúng hai fold v1 kém nhất. Bỏ màu thì gần như mọi
  style đã gặp (≤ 2,3 %), nhưng bảng tra bỏ màu chỉ được micro 0,439 (P 0,15): độ dày + kiểu không đủ phân biệt. Hướng: augment
  màu (v2 đã có hoán vị kênh / xám) và dựa vào hình học + ngữ cảnh.
- **F13. Ở fold lệch style, v1 cho điểm thấp cho ống thật: ngưỡng tốt nhất chạm đáy lưới 0,05.** Fold 4: cả 6 lần chấm chọn
  t = 0,05, F2@0,5 chỉ 0,42–0,45 (so với 0,54–0,60 ở ngưỡng chọn); fold 3: 3 / 6 lần 0,05. Fold 0–2 ngưỡng 0,10–0,30. Ngưỡng tối ưu
  có thể dưới 0,05 (lưới không xét); trong sản phẩm ngưỡng phải cố định trước, nên F2@0,5 của fold lệch style là con số thật hơn.
  Fold 4 còn giảm đều từ ep 5 (0,599 → 0,539 ở ep 25): học thuộc style của train.
- **F14. Model không biết trang là FIRE hay HVAC, dù nhãn định nghĩa khác nhau** (FIRE: nét dọc `Line`; HVAC: nét trong polygon
  ống gió) và AI_Takeoff luôn biết luồng. Cả v1 lẫn v2 không có đầu vào mode (`features.py`, `models/openstroke2.py`). Bằng chứng
  hiện chỉ yếu: thêm mode vào bảng tra bỏ màu tăng micro 0,439 → 0,472 (fold 2 +0,06, fold 0 / 1 −0,02 / −0,05). Ứng viên cho một
  thí nghiệm một biến sau E3, không đổi v2 đã ghi trước.

## 4. Lỗi đã gặp

- 10-20: bộ chuyển bản đầu xoay GT của doc 41 sai (F3). Đã sửa trước khi xuất dataset; test `test_calibration_offpage_points_count_as_misses`.
- 10-20: synthetic test đầu tiên đặt ống đối xứng nên `flip_y` trùng `identity`; test chốt toạ độ phải dùng hình bất đối xứng.

## 5. Hướng bị bác bỏ

(chưa có)

## 6. Hướng mở (xếp theo bằng chứng)

0. **E3: v2 (`docs/OPENSTROKE2.md`) hai hạt giống A (cửa sổ 65536) / B (16384), cùng fold.** Log v1 5 fold (E2-v1) chạy bằng trainer
   cũ trước `ed0c90c`: trên Colab chạy lại ô cài của notebook (đã `--force-reinstall`) để lấy bản trên `main`, rồi `openstroke bench`
   để chốt batch. Mốc của H7 = v1 micro 0,714 (tạm). Đọc kết quả theo fold, nhất là fold 3 / 4 (F12, F13).
1. E1.3 GBDT trên Colab CPU, cùng fold: cần để chấm H1 của v1 và làm mốc cho v2.
2. User quyết mục 3 AGENTS.md (cặp trùng, fold, mẫu âm, ống nước HVAC, GT phủ đủ trang cho các trang F4).
3. Sau E3: thêm mode FIRE / HVAC làm đầu vào (F14), một biến, zero-init.
4. Hạt giống B của v1 (`layer_pool=false`) chỉ khi còn cần chấm H4 / H5 cho v1; v2 cũng có LayerPool nên có thể chấm ở v2.
5. E1.2 cascade hiện tại từ run artifacts AI_Takeoff.
