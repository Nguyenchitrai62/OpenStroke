# E0 QA: nhãn true/other theo shape

Sinh tự động bởi `py -3.12 -m openstroke qa-e0` từ `datasets/e0/manifest.json` (2026-10-20T10:58:00, schema `openstroke.e0.v1`). Không sửa tay; chạy lại lệnh để cập nhật.

Quy tắc: FIRE `Line` trong 2.0 pt; HVAC polygon khớp `^(Duct|Elbow|Concentric Reducer|Eccentric Reducer|Chân rẽ)` (nới 2.0 pt để giữ thành ống trên viền); shape `true` khi ≥ 50% chiều dài trong vùng; chỉ group primary, reference; bezier lấy mẫu dọc cung. Nguồn: `F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3` (chỉ đọc, 2026-10-20T10:52:32).

**36 trang vật lý** (bỏ trùng theo sha256 + trang), trong đó 5 trang có hai GT → **41 bản GT** được gán nhãn, 41 đạt chốt toạ độ.

## 1. Bảng theo trang

Cột: *true (shape)* và *true (dài)* = tỷ lệ nét true theo số shape và theo chiều dài; *phủ GT* = phần chiều dài đường/viền GT nằm trong 2 pt của nét true; *GT trống* = số object GT phủ < 20 %; *chốt* = phép biến đổi được chọn, độ phủ của nó và lift so với runner-up (tolerance 0,5 pt); *nghi* = nét other cùng style với nét true nhưng cách mọi GT > 25 pt, theo % chiều dài true; *trộn* = shape có 10–90 % chiều dài trong vùng. ⚠ = tỷ lệ true lệch quá 3× trung vị của hệ (FIRE 10.5 %, HVAC 15.0 %).

| GT | File | Hệ | Shape | true (shape) | true (dài) | phủ GT | GT trống | chốt | nghi | trộn | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|
| doc000020_p0001_gt10 | 333.pdf | fire | 95,573 | 6.0 % | 12.9 % | 98.0 % | 0/349 | identity 96.1 %, lift 2.43 / 1.56 | 2.9 % | 1,348 | GT trùng, chờ user |
| doc000033_p0001_gt33 | 333.pdf | fire | 95,573 | 4.3 % | 11.3 % | 100.0 % | 0/132 | identity 100.0 %, lift 2.54 / 1.13 | 3.4 % | 1,163 | GT trùng, chờ user |
| doc000034_p0001_gt88 | L30M-LUMI.pdf | fire | 157,329 | 23.6 % | 9.7 % | 100.0 % | 0/203 | identity 100.0 %, lift 3.50 / 1.43 | 0.6 % | 2,225 |  |
| doc000035_p0001_gt83 | 7. FP_page 16.pdf | fire | 243,547 | 4.0 % | 8.1 % | 100.0 % | 0/268 | identity 100.0 %, lift 4.68 / 1.18 | 12.2 % | 1,130 |  |
| doc000036_p0001_gt89 | TOWER Z35.1 (S2) [39].pdf | fire | 56,729 | 3.9 % | 3.3 % ⚠ | 100.0 % | 0/121 | identity 100.0 %, lift 5.29 / 1.33 | 0.0 % | 777 |  |
| doc000037_p0001_gt87 | FP - Tang dien hinh 7-14, 16-33-Thap … | fire | 87,469 | 3.0 % | 6.6 % | 100.0 % | 0/127 | identity 91.0 %, lift 6.37 / 1.34 | 0.2 % | 1,319 |  |
| doc000060_p0001_gt86 | 0875-FP-CT2-C2Z1-303.pdf | fire | 246,327 | 3.0 % | 8.2 % | 100.0 % | 0/161 | identity 100.0 %, lift 8.00 / 1.77 | 0.9 % | 1,230 |  |
| doc000062_p0001_gt85 | 02.DN-TT-FP-BLOCK A_R1-Model.pdf | fire | 152,255 | 8.6 % | 13.1 % | 100.0 % | 0/801 | identity 100.0 %, lift 5.04 / 1.31 | 1.1 % | 6,641 |  |
| doc000063_p0001_gt84 | 2. HỆ THỐNG CHỮA CHÁY - BLOCK A - Bas… | fire | 176,430 | 10.1 % | 16.4 % | 100.0 % | 0/789 | identity 100.0 %, lift 5.71 / 1.62 | 0.5 % | 6,118 |  |
| doc000063_p0002_gt84 | 2. HỆ THỐNG CHỮA CHÁY - BLOCK A - Bas… | fire | 155,690 | 9.6 % | 15.0 % | 100.0 % | 0/748 | identity 100.0 %, lift 5.67 / 1.76 | 0.5 % | 4,818 |  |
| doc000064_p0001_gt90 | U35.2.pdf | fire | 81,149 | 12.7 % | 8.6 % | 100.0 % | 0/349 | identity 100.0 %, lift 3.18 / 1.29 | 0.3 % | 1,271 |  |
| doc000066_p0001_gt91 | 2.FIRE FIGHTING PLANS - FF MB REV-07.… | fire | 8,536 | 11.6 % | 12.6 % | 100.0 % | 0/199 | identity 100.0 %, lift 37.74 / 3.52 | 3.9 % | 1,102 |  |
| doc000022_p0001_gt11 | 20. 9062_M-ACB1----304.1 (DUCTING LAY… | hvac | 105,127 | 21.2 % | 24.7 % | 99.1 % | 0/316 | identity 100.0 %, lift 3.15 / 1.46 | 9.1 % | 1,415 |  |
| doc000023_p0001_gt12 | MB TH4.pdf | hvac | 160,713 | 26.5 % | 20.1 % | 100.0 % | 0/67 | identity 90.5 %, lift 3.06 / 0.99 | 5.1 % | 881 | GT trùng, chờ user |
| doc000025_p0001_gt14 | NOBU [24].pdf | hvac | 104,188 | 17.0 % | 20.0 % | 100.0 % | 0/102 | identity 100.0 %, lift 6.19 / 1.85 | 0.8 % | 744 | GT trùng, chờ user |
| doc000026_p0001_gt13 | NOBU [23].pdf | hvac | 72,475 | 12.7 % | 15.5 % | 100.0 % | 0/82 | identity 100.0 %, lift 5.18 / 1.33 | 1.3 % | 498 | GT trùng, chờ user |
| doc000030_p0001_gt15 | DC4-MEP-MV [40].pdf | hvac | 13,272 | 31.0 % | 15.0 % | 100.0 % | 0/159 | identity 100.0 %, lift 9.05 / 1.56 | 9.6 % | 685 |  |
| doc000038_p0001_gt26 | Y1_10-20.pdf | hvac | 72,131 | 1.7 % | 3.6 % ⚠ | 100.0 % | 0/27 | identity 100.0 %, lift 6.58 / 2.28 | 1.9 % | 101 |  |
| doc000038_p0002_gt26 | Y1_10-20.pdf | hvac | 64,707 | 0.7 % | 1.3 % ⚠ | 100.0 % | 0/16 | identity 100.0 %, lift 5.14 / 1.53 | 9.8 % | 42 |  |
| doc000038_p0003_gt26 | Y1_10-20.pdf | hvac | 61,238 | 1.0 % | 3.2 % ⚠ | 100.0 % | 0/20 | identity 100.0 %, lift 9.57 / 2.33 | 13.4 % | 63 |  |
| doc000038_p0004_gt26 | Y1_10-20.pdf | hvac | 96,733 | 1.2 % | 2.4 % ⚠ | 100.0 % | 0/29 | identity 100.0 %, lift 5.48 / 1.88 | 4.4 % | 107 |  |
| doc000038_p0005_gt26 | Y1_10-20.pdf | hvac | 85,463 | 1.0 % | 3.1 % ⚠ | 100.0 % | 0/24 | identity 100.0 %, lift 6.87 / 1.95 | 2.1 % | 84 |  |
| doc000038_p0006_gt26 | Y1_10-20.pdf | hvac | 64,124 | 3.5 % | 6.9 % | 100.0 % | 0/45 | identity 100.0 %, lift 8.97 / 1.42 | 25.5 % | 223 |  |
| doc000038_p0007_gt26 | Y1_10-20.pdf | hvac | 96,571 | 1.2 % | 2.5 % ⚠ | 100.0 % | 0/29 | identity 100.0 %, lift 5.36 / 1.78 | 2.0 % | 104 |  |
| doc000038_p0008_gt26 | Y1_10-20.pdf | hvac | 86,124 | 1.1 % | 2.9 % ⚠ | 100.0 % | 0/24 | identity 100.0 %, lift 6.51 / 2.78 | 2.2 % | 79 |  |
| doc000038_p0009_gt26 | Y1_10-20.pdf | hvac | 73,886 | 1.4 % | 3.6 % ⚠ | 100.0 % | 0/24 | identity 100.0 %, lift 8.55 / 2.77 | 5.0 % | 201 |  |
| doc000038_p0010_gt26 | Y1_10-20.pdf | hvac | 90,962 | 2.2 % | 2.0 % ⚠ | 100.0 % | 0/46 | identity 100.0 %, lift 9.66 / 2.03 | 0.0 % | 224 |  |
| doc000040_p0001_gt18 | 05. CHI TIET CAN HO-M-11.001.pdf | hvac | 51,201 | 0.9 % | 2.3 % ⚠ | 100.0 % | 0/52 | identity 100.0 %, lift 4.71 / 1.74 | 21.3 % | 136 |  |
| doc000041_p0001_gt17 | 26 M-MH-2007.1_ACMV LAYOUT PLAN_7TH F… | hvac | 214,672 | 2.5 % | 2.6 % ⚠ | 100.0 % | 0/163 | identity 100.0 %, lift 3.66 / 1.83 | 11.1 % | 843 |  |
| doc000043_p0001_gt21 | H1_KS.pdf | hvac | 370,668 | 26.9 % | 30.6 % | 100.0 % | 0/797 | identity 100.0 %, lift 3.07 / 1.05 | 2.4 % | 5,826 |  |
| doc000044_p0001_gt24 | NOBU [24].pdf | hvac | 104,188 | 14.1 % | 17.8 % | 100.0 % | 0/95 | identity 100.0 %, lift 6.19 / 1.81 | 3.3 % | 432 | GT trùng, chờ user |
| doc000045_p0001_gt23 | NOBU [23].pdf | hvac | 72,475 | 12.2 % | 15.0 % | 100.0 % | 0/82 | identity 100.0 %, lift 5.26 / 1.70 | 2.4 % | 455 | GT trùng, chờ user |
| doc000046_p0001_gt20 | C3-CH01_BASEMENT [14].pdf | hvac | 388,877 | 21.7 % | 21.8 % | 100.0 % | 0/592 | identity 100.0 %, lift 3.75 / 1.14 | 0.9 % | 4,355 |  |
| doc000047_p0001_gt19 | 04. MECHANICAL [58].pdf | hvac | 36,747 | 25.2 % | 18.3 % | 100.0 % | 0/370 | identity 100.0 %, lift 4.13 / 1.21 | 3.0 % | 1,279 |  |
| doc000050_p0001_gt25 | TB1.pdf | hvac | 124,821 | 5.1 % | 11.9 % | 100.0 % | 0/309 | identity 100.0 %, lift 7.91 / 1.30 | 9.2 % | 1,192 |  |
| doc000051_p0001_gt27 | 2D.pdf | hvac | 38,798 | 42.3 % | 37.2 % | 100.0 % | 0/770 | identity 100.0 %, lift 9.27 / 1.88 | 3.3 % | 1,204 |  |
| doc000053_p0001_gt28 | Pages from 4. HỆ THỐNG TĂNG ÁP HÚT KH… | hvac | 146,656 | 11.5 % | 6.9 % | 100.0 % | 0/132 | identity 99.8 %, lift 7.15 / 1.41 | 4.7 % | 406 |  |
| doc000054_p0001_gt29 | 2. TAHK_HAM-H-06.pdf | hvac | 211,199 | 30.6 % | 15.6 % | 100.0 % | 0/415 | identity 100.0 %, lift 4.49 / 1.10 | 5.9 % | 2,714 | GT trùng, chờ user |
| doc000055_p0001_gt30 | 2. TAHK_HAM-H-06.pdf | hvac | 211,199 | 28.9 % | 16.0 % | 100.0 % | 0/404 | identity 100.0 %, lift 3.98 / 1.02 | 5.0 % | 2,798 | GT trùng, chờ user |
| doc000056_p0001_gt31 | MB TH4.pdf | hvac | 160,713 | 28.0 % | 21.5 % | 100.0 % | 0/136 | identity 93.8 %, lift 3.39 / 1.37 | 5.8 % | 895 | GT trùng, chờ user |
| doc000059_p0001_gt32 | HVAC-B2.pdf | hvac | 338,458 | 7.2 % | 15.1 % | 99.6 % | 0/565 | identity 100.0 %, lift 5.60 / 1.25 | 3.5 % | 2,113 |  |

- Tổng FIRE (mọi bản GT đạt): 125,920 / 1,556,607 shape true (8.1 %), chiều dài true 10.9 %.
- Tổng HVAC (mọi bản GT đạt): 564,792 / 3,718,386 shape true (15.2 %), chiều dài true 14.0 %.

## 2. GT không có nét bên dưới (phủ < 20 %)

Không có.

## 3. Nét `other` đáng ngờ (giống ống nhưng xa GT)

Style = (layer, màu nét, độ dày, kiểu s/f/fs). Style "giống ống" khi ≥ 50 % chiều dài của nó là true và có ≥ 20 shape true. Vùng là ô 200 pt gom các nét đáng ngờ, toạ độ display top-left (pt) để mở trên UI GT. Đây là nơi GT có thể thiếu ống (quy tắc "còn lại là other" sẽ dạy sai), hoặc là nét cùng style nhưng thật sự không phải ống (ký hiệu, chú thích, sơ đồ). Cần user xác nhận.

### doc000038_p0006_gt26: Y1_10-20.pdf

436 shape, 6,535 pt (= 25.5 % chiều dài true).

- style `3.Y1_MẶT BẰNG ĐIỀU HÒA THÔNG GIÓ|INNO-HÚT KHÓI | #a8a8a8 | w=0.42 | s`: 173 shape, 2,782 pt (style này 80% true)
- style `3.Y1_MẶT BẰNG ĐIỀU HÒA THÔNG GIÓ|INNO-HÚT KHÓI | #000000 | w=0.84 | s`: 114 shape, 2,336 pt (style này 60% true)
- style `3.Y1_MẶT BẰNG ĐIỀU HÒA THÔNG GIÓ|INNO-GIOTUOI CẤP BÙ | #000000 | w=0.84 | s`: 149 shape, 1,417 pt (style này 62% true)
- vùng lớn nhất: [1485.2, 995.4, 1568.3, 1151.0] (112 shape, 2,048 pt); [1674.4, 1031.9, 1801.1, 1080.3] (102 shape, 1,815 pt); [742.3, 1098.8, 804.4, 1126.1] (30 shape, 379 pt); [632.2, 806.5, 664.4, 827.6] (42 shape, 318 pt); [890.5, 1059.0, 958.1, 1084.8] (22 shape, 299 pt)

### doc000040_p0001_gt18: 05. CHI TIET CAN HO-M-11.001.pdf

8 shape, 1,358 pt (= 21.3 % chiều dài true).

- style `360D-ACMV-FA | #000000 | w=0.36 | s`: 8 shape, 1,358 pt (style này 52% true)
- vùng lớn nhất: [1377.4, 952.6, 1858.6, 1053.8] (2 shape, 750 pt); [1406.1, 952.5, 1604.1, 1053.8] (2 shape, 324 pt); [1215.6, 937.2, 1331.4, 968.4] (4 shape, 284 pt)

### doc000038_p0003_gt26: Y1_10-20.pdf

23 shape, 1,178 pt (= 13.4 % chiều dài true).

- style `3.Y1_MẶT BẰNG ĐIỀU HÒA THÔNG GIÓ|INNO-HÚT KHÓI | #000000 | w=0.84 | s`: 10 shape, 1,008 pt (style này 60% true)
- style `3.Y1_MẶT BẰNG ĐIỀU HÒA THÔNG GIÓ|INNO-HÚT KHÓI | #a8a8a8 | w=0.42 | s`: 13 shape, 170 pt (style này 97% true)
- vùng lớn nhất: [375.3, 857.4, 750.9, 968.5] (2 shape, 647 pt); [257.9, 1414.5, 283.7, 1430.1] (14 shape, 253 pt); [608.5, 947.6, 752.8, 968.5] (4 shape, 176 pt); [991.8, 788.6, 1005.3, 807.4] (2 shape, 92 pt); [375.3, 856.7, 379.1, 866.4] (1 shape, 10 pt)

### doc000035_p0001_gt83: 7. FP_page 16.pdf

853 shape, 7,647 pt (= 12.2 % chiều dài true).

- style ` | #000000 | w=0.00 | fs`: 850 shape, 7,638 pt (style này 75% true)
- style ` | #000000 | w=0.31 | s`: 1 shape, 5 pt (style này 88% true)
- style ` | #000000 | w=0.42 | s`: 2 shape, 4 pt (style này 99% true)
- vùng lớn nhất: [409.6, 427.2, 601.3, 600.2] (212 shape, 2,192 pt); [1458.6, 1426.1, 1510.0, 1435.8] (159 shape, 1,676 pt); [1204.5, 1431.0, 1215.1, 1457.7] (72 shape, 763 pt); [136.5, 1410.8, 151.1, 1554.1] (29 shape, 601 pt); [137.7, 1335.3, 144.7, 1398.3] (16 shape, 296 pt)

### doc000041_p0001_gt17: 26 M-MH-2007.1_ACMV LAYOUT PLAN_7TH FLOOR - PART 1.pdf

3,295 shape, 3,939 pt (= 11.1 % chiều dài true).

- style `M-_ACONRETN_E0_N_ | #000000 | w=0.96 | s`: 1,822 shape, 1,542 pt (style này 52% true)
- style `M-_ACONRETN_E0_N_ | #000000 | w=0.12 | s`: 808 shape, 1,204 pt (style này 56% true)
- style `M-_ACONREDF_E0_N_ | #000000 | w=0.12 | s`: 181 shape, 796 pt (style này 56% true)
- vùng lớn nhất: [957.0, 860.6, 996.7, 902.8] (1230 shape, 1,359 pt); [491.6, 1145.7, 592.6, 1180.5] (402 shape, 921 pt); [1405.0, 459.8, 1439.4, 508.2] (277 shape, 630 pt); [1296.2, 466.2, 1356.6, 493.5] (155 shape, 403 pt); [1108.2, 353.4, 1116.4, 395.3] (535 shape, 197 pt)

### doc000038_p0002_gt26: Y1_10-20.pdf

20 shape, 494 pt (= 9.8 % chiều dài true).

- style `3.Y1_MẶT BẰNG ĐIỀU HÒA THÔNG GIÓ|INNO-HÚT KHÓI | #000000 | w=0.84 | s`: 7 shape, 324 pt (style này 72% true)
- style `3.Y1_MẶT BẰNG ĐIỀU HÒA THÔNG GIÓ|INNO-HÚT KHÓI | #a8a8a8 | w=0.42 | s`: 13 shape, 170 pt (style này 93% true)
- vùng lớn nhất: [257.9, 1416.3, 283.7, 1431.9] (14 shape, 253 pt); [1571.0, 945.5, 1598.3, 958.7] (3 shape, 121 pt); [935.5, 919.7, 962.8, 932.9] (3 shape, 121 pt)

### doc000030_p0001_gt15: DC4-MEP-MV [40].pdf

182 shape, 7,741 pt (= 9.6 % chiều dài true).

- style `M-AC-CSEA | #000000 | w=0.84 | s`: 27 shape, 3,120 pt (style này 77% true)
- style `M-AC-FA | #000000 | w=0.84 | s`: 17 shape, 2,515 pt (style này 79% true)
- style `M-AC-FA | #a6a6a6 | w=0.24 | s`: 138 shape, 2,106 pt (style này 84% true)
- vùng lớn nhất: [1892.9, 1168.0, 2031.5, 1441.4] (7 shape, 1,108 pt); [1894.1, 1137.8, 2009.2, 1168.0] (40 shape, 932 pt); [1799.7, 635.8, 2031.5, 778.8] (2 shape, 782 pt); [1746.5, 452.1, 2031.5, 472.0] (5 shape, 736 pt); [533.0, 966.2, 565.7, 983.0] (38 shape, 698 pt)

### doc000050_p0001_gt25: TB1.pdf

273 shape, 6,149 pt (= 9.2 % chiều dài true).

- style `M-O-DUCT-50 | #000000 | w=0.30 | s`: 192 shape, 2,836 pt (style này 56% true)
- style `M-O-DUCT-50 | #000000 | w=0.72 | s`: 28 shape, 1,632 pt (style này 82% true)
- style `M-E-DUCT-50 | #000000 | w=0.72 | s`: 8 shape, 665 pt (style này 96% true)
- vùng lớn nhất: [465.5, 332.2, 598.9, 418.8] (109 shape, 2,170 pt); [429.7, 389.4, 582.1, 474.5] (44 shape, 790 pt); [1515.7, 1505.4, 1573.0, 1523.1] (24 shape, 609 pt); [1403.0, 472.1, 1453.4, 568.3] (27 shape, 427 pt); [944.5, 552.0, 1079.3, 569.7] (2 shape, 336 pt)

### doc000022_p0001_gt11: 20. 9062_M-ACB1----304.1 (DUCTING LAYOUT PLAN – APT BASEMENT 1 ).pdf

1,796 shape, 16,429 pt (= 9.1 % chiều dài true).

- style `M-_MEVSPRESS-E0_N_ | #000000 | w=0.18 | s`: 677 shape, 4,026 pt (style này 70% true)
- style `M-_MEVSFRES_E0_N_ | #000000 | w=0.18 | s`: 306 shape, 2,147 pt (style này 71% true)
- style `M-_MEVSPRESS-E0_N_ | #000000 | w=1.02 | s`: 48 shape, 2,088 pt (style này 52% true)
- vùng lớn nhất: [1362.7, 399.9, 1536.9, 524.9] (399 shape, 3,710 pt); [1239.5, 527.3, 1364.5, 881.1] (375 shape, 2,913 pt); [1596.5, 1022.4, 1799.5, 1248.6] (172 shape, 1,994 pt); [633.7, 660.4, 651.3, 955.1] (119 shape, 1,525 pt); [1469.6, 309.1, 1569.6, 482.0] (109 shape, 1,509 pt)

### doc000054_p0001_gt29: 2. TAHK_HAM-H-06.pdf

5,113 shape, 10,434 pt (= 5.9 % chiều dài true).

- style `INNO-ACMV-B.EA | #bababa | w=0.00 | s`: 202 shape, 4,396 pt (style này 78% true)
- style `INNO-ACMV-B.EA | #000000 | w=0.00 | s`: 3,847 shape, 4,034 pt (style này 60% true)
- style `TA | #000000 | w=0.00 | s`: 85 shape, 969 pt (style này 67% true)
- vùng lớn nhất: [1622.5, 1391.5, 1796.8, 1471.2] (99 shape, 2,153 pt); [1216.2, 217.5, 1389.9, 354.4] (149 shape, 695 pt); [1607.6, 1232.3, 1728.9, 1408.0] (37 shape, 604 pt); [51.9, 885.2, 191.3, 954.1] (2 shape, 556 pt); [51.9, 1556.2, 191.3, 1625.1] (2 shape, 556 pt)

### doc000056_p0001_gt31: MB TH4.pdf

1,418 shape, 5,212 pt (= 5.8 % chiều dài true).

- style `CDC-ACMV-SE (EI) | #333333 | w=0.12 | s`: 1,128 shape, 2,664 pt (style này 94% true)
- style `CDC-ACMV-PA | #000000 | w=0.72 | s`: 40 shape, 931 pt (style này 66% true)
- style `CDC-ACMV-B.SA | #000000 | w=0.54 | s`: 38 shape, 918 pt (style này 65% true)
- vùng lớn nhất: [1199.1, 170.7, 1226.0, 245.5] (446 shape, 1,159 pt); [1312.6, 898.6, 1461.2, 1039.9] (107 shape, 1,012 pt); [300.5, 160.0, 389.8, 187.0] (288 shape, 718 pt); [305.1, 697.4, 332.5, 744.8] (94 shape, 597 pt); [1199.1, 172.1, 1228.2, 202.1] (262 shape, 573 pt)

### doc000023_p0001_gt12: MB TH4.pdf

1,380 shape, 4,294 pt (= 5.1 % chiều dài true).

- style `CDC-ACMV-SE (EI) | #333333 | w=0.12 | s`: 1,128 shape, 2,664 pt (style này 91% true)
- style `CDC-ACMV-PA | #000000 | w=0.72 | s`: 40 shape, 931 pt (style này 66% true)
- style `CDC-ACMV-B.EA | #000000 | w=0.72 | s`: 86 shape, 482 pt (style này 65% true)
- vùng lớn nhất: [1312.6, 898.6, 1461.2, 1039.9] (107 shape, 1,012 pt); [1199.1, 199.3, 1226.0, 245.5] (443 shape, 780 pt); [300.5, 160.0, 389.8, 187.0] (288 shape, 718 pt); [1199.3, 175.0, 1226.0, 200.6] (248 shape, 395 pt); [674.9, 338.6, 726.8, 373.7] (16 shape, 339 pt)

## 4. Năm PDF có hai GT trên cùng trang

Hai bản GT được gán nhãn riêng (`npz` của cả hai đều có). *IoU dài* = chiều dài nét true của cả hai / hợp. Chưa chọn bản nào: quyết định của user (AGENTS.md mục 3.1).

| File | Bản A | Bản B | Object trong phạm vi A / B | Dài true A / B (pt) | Chỉ A true | Chỉ B true | IoU dài |
|---|---|---|---|---|---|---|---|
| 333.pdf | doc000020_p0001_gt10 | doc000033_p0001_gt33 | 349 / 132 | 29,950 / 25,183 | 1,669 shape, 3,627 pt | 46 shape, 533 pt | 0.864 |
| MB TH4.pdf | doc000023_p0001_gt12 | doc000056_p0001_gt31 | 67 / 136 | 83,517 / 89,272 | 0 shape, 0 pt | 2,422 shape, 5,755 pt | 0.936 |
| NOBU [24].pdf | doc000025_p0001_gt14 | doc000044_p0001_gt24 | 102 / 95 | 74,491 / 66,151 | 3,234 shape, 9,585 pt | 212 shape, 1,245 pt | 0.857 |
| NOBU [23].pdf | doc000026_p0001_gt13 | doc000045_p0001_gt23 | 82 / 82 | 41,893 / 40,536 | 459 shape, 1,931 pt | 139 shape, 574 pt | 0.941 |
| 2. TAHK_HAM-H-06.pdf | doc000054_p0001_gt29 | doc000055_p0001_gt30 | 415 / 404 | 176,074 / 181,290 | 6,497 shape, 3,750 pt | 2,937 shape, 8,966 pt | 0.931 |

- 333.pdf: hai document có payload khác nhau; bản B được gán nhãn lại trên payload của A để so cùng nét (dài true của B trên payload A: 26,856 pt).
- 333.pdf: A {'Line': 349} · B {'Line': 132}
- MB TH4.pdf: A {'Chân rẽ': 1, 'Concentric Reducer': 9, 'Duct': 39, 'Eccentric Reducer': 4, 'Elbow': 3, 'Elbow 45': 3, 'Elbow 90': 8} · B {'Chân rẽ': 31, 'Concentric Reducer': 8, 'Duct': 69, 'Eccentric Reducer': 4, 'Elbow': 6, 'Elbow 90': 18}
- NOBU [24].pdf: A {'Chân rẽ': 4, 'Concentric Reducer': 9, 'Duct': 61, 'Eccentric Reducer': 7, 'Elbow 90': 21} · B {'Chân rẽ': 4, 'Chân rẽ vuông tròn': 9, 'Concentric Reducer': 4, 'Duct': 56, 'Elbow 90': 22}
- NOBU [23].pdf: A {'Chân rẽ': 2, 'Concentric Reducer': 5, 'Duct': 51, 'Eccentric Reducer': 6, 'Elbow 90': 18} · B {'Chân rẽ': 2, 'Chân rẽ vuông tròn': 6, 'Concentric Reducer': 1, 'Duct': 53, 'Elbow 90': 20}
- 2. TAHK_HAM-H-06.pdf: A {'Chân rẽ': 10, 'Concentric Reducer': 10, 'Duct': 229, 'Eccentric Reducer': 53, 'Eccentric Reducer vuông tròn': 1, 'Elbow': 35, 'Elbow 90': 77} · B {'Chân rẽ': 12, 'Concentric Reducer': 8, 'Duct': 224, 'Eccentric Reducer': 53, 'Elbow': 26, 'Elbow 90': 74, 'Elbow vuông': 7}

