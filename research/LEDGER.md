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

## 4. Lỗi đã gặp

- 10-20: bộ chuyển bản đầu xoay GT của doc 41 sai (F3). Đã sửa trước khi xuất dataset; test `test_calibration_offpage_points_count_as_misses`.
- 10-20: synthetic test đầu tiên đặt ống đối xứng nên `flip_y` trùng `identity`; test chốt toạ độ phải dùng hình bất đối xứng.

## 5. Hướng bị bác bỏ

(chưa có)

## 6. Hướng mở (xếp theo bằng chứng)

0. **E3: v2 (`docs/OPENSTROKE2.md`) hai hạt giống A (cửa sổ 65536) / B (16384), cùng mốc v1 + GBDT trên cùng fold.** Trước đó chạy
   `openstroke bench` trên G4 để chốt batch.
1. User quyết mục 3 AGENTS.md (cặp trùng, fold, mẫu âm, ống nước HVAC, GT phủ đủ trang cho các trang F4).
2. E1.1 trần chọn layer (local) và E1.3 cây (Colab), cùng fold.
3. E2 model v1, hai hạt giống A / B (`docs/OPENSTROKE1.md`), tiêu chí ở `HYPOTHESES.md`.
4. E1.2 cascade hiện tại từ run artifacts AI_Takeoff.
