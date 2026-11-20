# Nghiên cứu kiến trúc model bắt nét HVAC

## F207. Tín hiệu hình học còn thiếu sau E130

E114 là đối chứng ảnh + vector không layer. Model trả một xác suất trên shape của PDF. E129 giảm F2 nét Viewer từ 0,768 xuống 0,752. E130 giữ 0,768 và giảm AP, recall. E130 đọc một lượt hàng xóm có xác suất E114 ≥ 0,20 rồi sửa logit từ embedding đã đóng băng. Trong E114, ShapeEncoder đọc 16 mẫu độ dài cung qua PointNet, nhánh ảnh lấy trung bình đặc trưng FPN tại các mẫu, và các shape được xếp cửa sổ theo tâm.

CADSpotting dùng điểm dày theo đường primitive rồi gộp về primitive. GAT-CADNet mã hóa hình học tương đối giữa primitive. Hướng xét là giữ nhiều vị trí trên một nét đủ lâu để đọc ngữ cảnh ngang qua vách và dọc theo đầu nối, rồi gộp về một xác suất trên shape để chạy cùng hậu kỳ đối chứng. Không dùng layer, tập CAD ngoài, bộ giải mã instance hay mã và trọng số của các bài. [CADSpotting](https://www.sciopen.com/article/10.26599/CVM.2026.9450573?issn=2096-0433) · [GAT-CADNet](https://arxiv.org/abs/2201.00625)

Đăng ký probe trước. E114 TTA ×8, ngưỡng 0,20, cùng 13 trang e12 hvac10b phát triển. Dùng đúng payload IDs đã chấm. Với mỗi kiến trúc giả định, dựng tập nét có tín hiệu mà nó có thể khai thác và sửa dự đoán lý tưởng chỉ trong tập đó. Trần cơ hội F2 nét gồm ba nhánh độc lập: (A) vách song song có chồng chiều dài ít nhất 50% trên vách ngắn, tính trên cạnh đối diện kiểu 8 của E130, không cổng E114; (B) primitive thuộc một object mà các primitive đồng object không vào trọn bất kỳ cửa sổ 65.536 token với stride 50%; (C) shape có điểm gián đoạn bút trên 1 pt bị gộp trong cùng token. Điều kiện (B) chỉ dùng nhãn làm oracle để đo giới hạn của cửa sổ. Đo train–val AI_Takeoff riêng trên điểm TTA ×1 cùng ngưỡng, macro theo PDF. Phiên này không chạy model hay fit thêm.

Chỉ xét nhánh cho trần tăng ≥0,10 trên Viewer và chẩn đoán train–val phù hợp để mở version mới. Tiêu chí cuối là F2 object trên dữ liệu giữ ngoài huấn luyện, so với E114 qua cùng hậu kỳ. 8 trang Viewer có nhãn một phần, chỉ dùng chẩn đoán.

### Kết quả F207

| Sửa lý tưởng theo vùng có tín hiệu | GT5 F2 nét | Viewer8 F2 nét | Δ Viewer8 |
|---|---:|---:|---:|
| E114 tại 0,20 | 0,9112 | 0,7761 | — |
| A. Có đoạn song song đối diện | 0,9267 | 0,8482 | +0,0721 |
| B. Object không vào trọn cửa sổ | 0,9113 | 0,7788 | +0,0027 |
| C. Shape có các đường con rời | 0,9124 | 0,7764 | +0,0003 |
| Hợp A, B và C | 0,9280 | 0,8485 | +0,0724 |

A cao nhất ở vw_2f3d p4 (0,7467 → 0,9420) và vw_579f p2 (0,7260 → 0,9169). Hai trang vw_62b1 chỉ tăng 0,0296 và 0,0465. Đồ thị đối diện hiện tại không đủ bao phủ để xem một nhánh chỉ nhận cặp vách như giải pháp chung.

TTA ×1, ngưỡng 0,20, riêng AI_Takeoff. E114 trên 19 trang train đã thấy, gom 10 PDF: F2/P/R micro 0,9259/0,7941/0,9659, mean PDF 0,9429, PDF thấp nhất 0,7673. GT5 validation: 0,9097/0,8271/0,9329, mean PDF 0,9056, thấp nhất 0,8848. Chênh lệch mean PDF 0,0373, thua chính ở recall. Đồng thời precision tổng train thấp hơn validation. Số này không đủ để quy toàn bộ lỗi cho thiếu capacity hoặc suy giảm precision trên PDF lạ. Mở rộng window và xử lý đường con rời là ưu tiên thấp.

## F208. Kiểm tra khái quát hóa của ngữ cảnh vách và đầu nối

Đặt trước tiêu chí. Trên điểm TTA ×1 của GT train và GT val, cùng nhóm điểm 0,01 ≤ p < 0,20, đo tỷ lệ chiều dài dương khi có cặp vách song song và khi hai vách có cùng láng giềng chạm đầu hoặc giao T. Đặc trưng chỉ từ hình học không layer. Đo chênh tỉ lệ dương so với nhóm không có tín hiệu và tỉ lệ nét dương bị bỏ sót có tín hiệu. Tín hiệu ứng viên phải cùng chiều, hơn ít nhất 15 điểm phần trăm ở cả hai split, và phủ ít nhất 10% chiều dài dương bị bỏ sót trên GT val. Báo thêm ở nhóm 0,20 ≤ p < 0,80 để xem nguy cơ gọi nền là ống. Đây là chẩn đoán phân biệt nét, không fit classifier và không đổi kết quả F207.

### Kết quả F208

Tỷ lệ nét dương tính theo chiều dài cùng nhóm xác suất. Các cột hình học loại trừ lẫn nhau.

| Split, xác suất E114 | Không có vách đối diện | Có vách, chưa thấy đầu nối chung | Có vách và đầu nối chung |
|---|---:|---:|---:|
| GT train, 0,01–0,20 | 2,6% | 6,0% | 4,9% |
| GT val, 0,01–0,20 | 3,9% | 6,4% | 4,7% |
| GT train, 0,20–0,80 | 25,5% | 25,8% | 56,7% |
| GT val, 0,20–0,80 | 38,6% | 18,4% | 11,0% |

Tín hiệu vách phủ 18,8% chiều dài nét dương bị bỏ sót ở GT val, và tín hiệu vách có đầu nối chung phủ 7,3%. Mức phân biệt ở dải điểm thấp thiếu xa mốc 15 điểm phần trăm. Ở dải điểm giữa, quan hệ vách + đầu nối chung đổi dấu khi qua PDF validation. Bác bỏ việc dùng riêng motif vách/đầu nối để quyết định nét hoặc làm cổng cứng. Đồ hình tương tự xuất hiện trong nét nền; cần thêm ngữ cảnh khác để phân biệt.

## Kiến trúc ứng viên, thiết kế chưa triển khai

**Giữ biểu diễn đoạn có hướng đến bước hợp nhất ảnh và vector.** E114 thực ra đã lấy mẫu dọc nét, nhưng gộp ảnh theo trung bình và vector theo max/mean trước khi thân mạng đọc ngữ cảnh. Nghiên cứu tiếp nhánh lấy mẫu FPN có hướng theo từng đoạn tại chỗ, rồi gộp về đúng shape. Từ VecFormer lấy ý tưởng giữ các đoạn liên tục cùng hình học đoạn. Từ GAT-CADNet lấy mã hóa tương đối. Từ CADSpotting lấy điểm đặt mẫu theo độ dài cung. Giữ CNN và đầu nhị phân E114. Các kỹ thuật phân instance ở ba bài và việc nhân toàn bộ point cloud qua PTv3 nằm ngoài ứng viên này. [VecFormer, NeurIPS 2025](https://openreview.net/pdf/85479f6eb8dde3e260ef124bc0eb56154e4e1e87.pdf) · [CADSpotting](https://www.sciopen.com/article/10.26599/CVM.2026.9450573?issn=2096-0433) · [GAT-CADNet](https://arxiv.org/abs/2201.00625)

1. Lấy vài vị trí cách đều độ dài cung trên từng nét, giữ tiếp tuyến và vị trí chuẩn hóa trên nét. Với đường cong dùng tiếp tuyến cục bộ, dùng dấu ngắt giữa các đoạn rời. Lấy đặc trưng FPN tại nét và hai phía theo pháp tuyến ở vài mức cự ly.
2. MLP dùng chung đọc các vai trò dọc và ngang tại từng vị trí. Gộp hai phía đối xứng khi đảo chiều nét, rồi mới pool có học về shape. Dùng thông tin vùng ảnh hai phía để phân biệt ống với nét kiến trúc có cùng màu, độ dày hoặc cấu trúc vách.
3. Projection zero-init đưa kết quả vào token trước các block hợp nhất của E114. Mọi shape được đọc ngữ cảnh bất kể điểm E114. Một logit mỗi shape, không có cổng điểm theo hàng xóm. Chi phí lấy mẫu tăng tuyến tính theo số shape và số vị trí, cần đo bộ nhớ FPN theo batch. Thử chiếu kênh trước khi lấy mẫu và gom theo khối để tránh giữ tensor lớn. Không thay kích thước cửa sổ.
4. Khi đủ bằng chứng để mở version, so E114 gốc, fine-tune đối chứng không nhánh mới, và fine-tune có nhánh mới với cùng mức mở trọng số, seed, optimizer, số bước và GT19. Ablation chỉ lấy mẫu trên nét để tách giá trị của hai phía. Chọn checkpoint trên GT val, xem F2/P/R nét từng PDF, sau đó chấm F2/P/R object cùng hậu kỳ theo lớp, trang, độ phủ object có nét. Yêu cầu cải thiện F2 object ≥0,02 ở pilot và không tụt recall trang quá 0,02 là tiêu chí nghiên cứu. Mục tiêu phát hành giữ F2 object ≥0,80 trên test dự án chưa thấy. E114 từng dùng các trang hiện tại để phát triển, nên chưa có số test kín.

F207 và F208 chưa đạt cổng nghiên cứu thường lệ. User sau đó yêu cầu mã để tự chạy Colab. Một pilot có giới hạn đã được triển khai thành E131. Điều kiện, ngân sách và lệnh train nằm trong [OpenStroke37](OPENSTROKE37.md). Chưa có số đo E131. E114 là đối chứng, dùng cùng e12.
