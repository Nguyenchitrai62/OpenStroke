# Kiểm toàn bộ GT HVAC (AI_Takeoff DB, chỉ đọc)

Remote: [('https://ai-takeoff.lm.io.vn', '2026-11-12 10:02:36', 'succeeded')]

- Trang GT HVAC có object primary/reference: **30**; thuộc tài liệu mirror từ remote (GT chính thức): **24** (15 tài liệu); chỉ có ở local: 6 (6 tài liệu).
- Trang có hai bản GT (cùng sha256 + trang): 4.
- Object 6 lớp trên trang remote: Duct 2684, Elbow 922, Elbow vuông 58, Concentric Reducer 413, Eccentric Reducer 184, Chân rẽ 693

## Trang remote (dùng để dựng dataset)

| doc | trang | GT | sửa lần cuối | file | object 6 lớp | ngoài phạm vi | < 3 đỉnh | payload | trong obb_hvac7 |
|---|---|---|---|---|---|---|---|---|---|
| 180 | 1 | 114 | 2026-08-26 10:32 | 04. MECHANICAL [58].pdf | 370 | 0 | 0 | **thiếu** | val 370 |
| 181 | 1 | 115 | 2026-08-20 04:44 | 05. CHI TIET CAN HO-M-11.001.pdf | 52 | 84 | 0 | **thiếu** | train 52 |
| 187 | 1 | 117 | 2026-08-28 05:23 | 2. TAHK_HAM-H-06.pdf | 404 | 0 | 0 | **thiếu** | val 404 |
| 189 | 1 | 118 | 2026-08-26 10:27 | 26 M-MH-2007.1_ACMV LAYOUT PLAN_7TH FLOO | 163 | 18 | 0 | **thiếu** | val 163 |
| 190 | 1 | 119 | 2026-11-12 10:06 | 2D.pdf | 840 | 131 | 0 | **thiếu** | val 827 (đổi) |
| 191 | 1 | 120 | 2026-08-27 07:24 | C3-CH01_BASEMENT [14].pdf | 592 | 408 | 0 | **thiếu** | train 592 |
| 194 | 1 | 121 | 2026-08-28 06:54 | H1_KS.pdf | 797 | 0 | 0 | **thiếu** | train 797 |
| 195 | 1 | 122 | 2026-08-28 05:27 | HVAC-B2.pdf | 565 | 0 | 0 | **thiếu** | train 565 |
| 196 | 1 | 123 | 2026-08-26 08:36 | MB TH4.pdf | 136 | 0 | 0 | **thiếu** | train 136 |
| 198 | 1 | 124 | 2026-08-28 07:06 | NOBU [23].pdf | 82 | 70 | 0 | **thiếu** | train 82 |
| 199 | 1 | 125 | 2026-08-28 07:05 | NOBU [24].pdf | 95 | 63 | 0 | **thiếu** | train 95 |
| 200 | 1 | 126 | 2026-08-26 08:08 | Pages from 4. HỆ THỐNG TĂNG ÁP HÚT KHÓI  | 132 | 0 | 0 | **thiếu** | val 132 |
| 201 | 1 | 127 | 2026-08-27 07:00 | TB1.pdf | 309 | 0 | 0 | **thiếu** | val 309 |
| 211 | 1 | 129 | 2026-10-28 03:17 | UAT_clean_B2_1-200_v1.pdf | 133 | 24 | 0 | **thiếu** | train 144 |
| 214 | 1 | 130 | 2026-08-27 07:48 | Y1_10-20.pdf | 27 | 0 | 0 | **thiếu** | train 27 |
| 214 | 2 | 130 | 2026-08-27 07:48 | Y1_10-20.pdf | 16 | 0 | 0 | **thiếu** | train 16 |
| 214 | 3 | 130 | 2026-08-27 07:48 | Y1_10-20.pdf | 20 | 0 | 0 | **thiếu** | train 20 |
| 214 | 4 | 130 | 2026-08-27 07:48 | Y1_10-20.pdf | 29 | 0 | 0 | **thiếu** | train 29 |
| 214 | 5 | 130 | 2026-08-27 07:48 | Y1_10-20.pdf | 24 | 0 | 0 | **thiếu** | train 24 |
| 214 | 6 | 130 | 2026-08-27 07:48 | Y1_10-20.pdf | 45 | 0 | 1 | **thiếu** | train 44 (đổi) |
| 214 | 7 | 130 | 2026-08-27 07:48 | Y1_10-20.pdf | 29 | 0 | 0 | **thiếu** | train 29 |
| 214 | 8 | 130 | 2026-08-27 07:48 | Y1_10-20.pdf | 24 | 0 | 0 | **thiếu** | train 24 |
| 214 | 9 | 130 | 2026-08-27 07:48 | Y1_10-20.pdf | 24 | 0 | 0 | **thiếu** | train 24 |
| 214 | 10 | 130 | 2026-08-27 07:48 | Y1_10-20.pdf | 46 | 10 | 0 | **thiếu** | train 46 |

## Trang chỉ có ở local (không dùng)

- doc 22 trang 1 GT 11: 20. 9062_M-ACB1----304.1 (DUCTING LAYOUT PLAN – APT BASEMENT 1 ).pdf
- doc 23 trang 1 GT 12: MB TH4.pdf
- doc 25 trang 1 GT 14: NOBU [24].pdf
- doc 26 trang 1 GT 13: NOBU [23].pdf
- doc 30 trang 1 GT 15: DC4-MEP-MV [40].pdf
- doc 54 trang 1 GT 29: 2. TAHK_HAM-H-06.pdf

## Trang có hai bản GT

- MB TH4.pdf trang 1: GT 12 (doc 23, sửa 2026-08-18 07:50, local), GT 123 (doc 196, sửa 2026-08-26 08:36, remote)
- NOBU [24].pdf trang 1: GT 14 (doc 25, sửa 2026-08-19 03:33, local), GT 125 (doc 199, sửa 2026-08-28 07:05, remote)
- NOBU [23].pdf trang 1: GT 13 (doc 26, sửa 2026-08-19 03:13, local), GT 124 (doc 198, sửa 2026-08-28 07:06, remote)
- 2. TAHK_HAM-H-06.pdf trang 1: GT 29 (doc 54, sửa 2026-08-25 14:25, local), GT 117 (doc 187, sửa 2026-08-28 05:23, remote)

## Tên object ngoài 6 lớp (bỏ qua khi dựng)

Line 521, Ống nước ngưng 98, VCD 88, Ống đồng 34, MD 23, Tee 20, Volume Control Damper 11, OBD 10, FD 3
