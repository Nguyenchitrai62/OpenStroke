# SOTA nhận diện nét vector CAD: ghi chú cho OpenStroke (2026-11-02)

Tổng hợp bởi subagent từ bài gốc. Số arXiv đọc qua bản tóm tắt tự động, **cần kiểm lại trước khi trích dẫn**. Hầu hết đo trên FloorPlanCAD (nhiều lớp, bản vẽ sạch, không có chuyện nét đứt bị bung), không so trực tiếp với bài toán này. Chưa tìm thấy bài vector-stroke cho MEP.

| Phương pháp | Nét → token → primitive | Layer / đồ thị / ảnh | Số báo cáo | License code |
|---|---|---|---|---|
| [VecFormer](https://arxiv.org/html/2505.23395v3) | đoạn ≤ α·min(W,H), 7 chiều; max + mean | LFE / không / không | PQ 91,1, PQst 90,4 | Apache-2.0 |
| [CADSpotting](https://arxiv.org/html/2412.07377v4) | điểm dày d = 0,14; max + mean | không | PQ 87,4, wF1 93,2 | chưa thấy repo |
| [SymPoint-V2](https://arxiv.org/html/2407.01928v1) | 1 điểm | LFE | PQ 90,1 | không ghi |
| [GAT-CADNet](https://arxiv.org/html/2201.00625) | 1 đỉnh, cạnh 7 chiều | đồ thị | PQ 73,7 | chưa thấy |
| [VectorGraphNET](https://arxiv.org/html/2410.01336v1) | nút path, 2 GATv2, 1,3M | kNN | wF1 89,0 | chưa thấy |
| [DPSS](https://arxiv.org/html/2503.22346v3) | 1 điểm + HRNet | ảnh | PQ 86,2 (có prior 89,5) | NOASSERTION |
| [TextCAD](https://arxiv.org/html/2607.12678v1) | đoạn VecFormer + chữ | không ảnh | PQ 91,09 so với VecFormer 88,14 | chưa thấy |

**So trong cùng bài (đáng tin hơn so giữa các bài):**
- CADSpotting, PQ lớp tường: đặc trưng tay 44,88 → lấy mẫu dày 62,55 → + pooling 73,21.
- VecFormer: điểm → đoạn PQst 85,4 → 85,9; layer id +1,8, LFE +0,9.
- DPSS: chỉ primitive 81,7, chỉ ảnh 80,9, nối 83,0, **hợp nhất có cổng 86,2**.
- GAT-CADNet: 0,602 → 0,737 nhờ đồ thị. SymPoint: ACM, CCL cộng +1,2; SPv2 bỏ vì hạn chế.
- CADTransformer: thêm chữ 0,7152 → 0,7352; random layer augmentation 0,6732 → 0,6894.
- Sonata so với PTv3 từ đầu: 1 % nhãn 25,8 → 45,3; 10 % 61,0 → 72,4; đủ nhãn 77,2 → 79,4.

Phần layer trong các bài mâu thuẫn với F19, F23, E12, E27 của repo này, nên không đề xuất thêm đầu vào layer.

## Áp vào OpenStroke (suy luận của agent, chưa đo)

1. **Vẽ đúng nét thật cho nhánh ảnh, đọc nét dài dày hơn, gộp max + mean.** `render` hiện nối 16 điểm lấy mẫu nên bỏ hai đầu nét và kẻ dây cung qua khe.
2. **Đọc lại ảnh muộn, có cổng** (kiểu DPSS), sau tầng giải mã 0; zero-init đúng.
3. **Tiền huấn luyện tự chưng cất mù kiểu nét** (Sonata / Utonia) thay MSE trên 47 đặc trưng, nhắm trang sập vì kiểu vẽ lạ.
4. **Token chữ** qua cross-attention zero-init, phân lớp bằng regex (cỡ ống, cao độ, ký hiệu hệ), nhắm FP là ống hệ khác.
5. **Token con cho nét dài**, chỉ tách nét > T: T = 100 thêm ~1 % token. Trần nhỏ (+0,017 FIRE) vì nét dài thiếu recall gần như chỉ ở 3 trang (F50).
