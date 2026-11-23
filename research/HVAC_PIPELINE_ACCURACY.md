# Độ chính xác OpenStroke HVAC

## F205. Chẩn đoán trần hậu kỳ và chọn nét

Đối chứng triển khai E114. F151 đo F2 object 0,882 trên GT5 và 0,593 trên Viewer8 với logic HVAC ở p ≥ 0,05, luật thin và vùng kín ở p ≥ 0,30. Viewer có nhãn một phần. Đây là kết quả lịch sử dùng rule_nl, 6 lớp và IoU nét ≥ 0,50. App hiện còn một cấu hình ưu tiên item và dùng ngưỡng từ checkpoint, nên cần kiểm riêng cấu hình đang chạy. E129 và E130 chưa vượt E114 trên Viewer về F2 nét. E130 giữ 0,768, recall và AP giảm.

Dùng điểm E114 đã lưu, đúng 13 trang hvac10b và hình học trong score pack đã đóng băng. Giữ các polygon logic đã cache ở 0,05. Probe ghi trước ba nhánh dựng vùng buffer F151: nét p ≥ 0,30, nét p ≥ 0,05 và mặt nạ đúng từ GT. So cả recall hình học của vùng và phần tăng khi gộp nguồn logic. Đối sánh một-một tại IoU diện tích ≥ 0,50, bỏ qua lớp, báo theo trang và lớp. Phép đo chẩn đoán nguồn đề xuất dùng IoU diện tích; F2 object sản phẩm dùng membership theo chiều dài. Không dùng nhãn Viewer để train hoặc chọn tham số.

Giả thuyết H147. Giữ mọi object đã dựng, thêm một số vùng kín cả hai loại nét mạnh và yếu có cùng đường bao để phục hồi đối tượng bị đứt nét. Cổng triển khai thử là vùng bổ sung có trần tăng recall hình học ≥ 0,10 trên cùng Viewer8. Nếu hạ ngưỡng và mặt nạ đúng đều không qua cổng, giữ phát sinh vùng như cũ và khoanh lỗi vào hình học hoặc lớp. Nếu chỉ mặt nạ đúng qua cổng, ưu tiên sửa khả năng chọn đúng nét ở model. Không đổi ổn định app trước khi chấm cùng rule_nl và protocol object hiện hành. Không chạy lại inference mạng, dùng điểm và GT đã lưu.

Kỳ vọng hậu kỳ trên GT validation, tăng F2 object ít nhất 0,02 hoặc Viewer tăng ít nhất 0,03, GT giảm tối đa 0,01 và báo đủ trang cùng lớp. Kết quả triển khai chưa đo.

Kết quả F205 từ điểm E114 TTA ×8. Các cột là số object có đề xuất khớp vị trí ở IoU diện tích ≥ 0,50. Mỗi ứng viên chỉ được ghép một lần.

| Nguồn đề xuất | GT5, 1.375 object | Viewer8, 2.009 object |
|---|---:|---:|
| Logic đã cache + vùng p ≥ 0,30 | 1.338 (97,3%) | 1.352 (67,3%) |
| Hợp thêm vùng p ≥ 0,05 | 1.341 (97,5%) | 1.428 (71,1%) |
| Hợp thêm vùng từ mask GT | 1.365 (99,3%) | 1.459 (72,6%) |
| Hợp thêm vùng từ đồ thị chạm đầu, cùng p ≥ 0,30 | 1.344 (97,7%) | 1.404 (69,9%) |

Mở ngưỡng tạo thêm 3.680 vùng Viewer trước xếp hạng để tăng trần 76 object. H147 không qua cổng. Hợp đề xuất đồ thị và buffer bổ sung 52 object Viewer trong chẩn đoán. Trong 437 Duct Viewer chưa được đề xuất khớp, 168 có ít nhất 60% mỗi vách dài đã được nét p ≥ 0,30 phủ. Trong số đó, 138 có ít nhất một đầu ống được phủ ≥75%, chỉ 14 có cả hai. Đây là probe dùng chính hộp xoay GT để xem phần vách có sẵn. Nó không thể dự đoán mức thu hồi của thuật toán tự tìm vách.

## H148. Khép đầu ống còn hở từ ba cạnh đang có

Thử tạo một nguồn đề xuất Duct bổ sung từ hai vách cùng hướng và một nét đóng đầu có thật. Chỉ đọc đoạn thẳng đã được model chọn tại 0,30. Nét đóng đầu dài 2–60 pt. Hai vách bắt đầu cách đầu nét ngang ≤1 pt và gần vuông góc với nó (±10°), cùng hướng trong 5°, phần chênh chiều dài ≤15% hoặc 2 pt, tỷ lệ dài/rộng ≥1,5. Tạo đa giác từ nét đóng đầu và hai đầu còn lại. Gộp trùng và bỏ vùng lấn vùng đã dựng. Biến thử là nguồn đề xuất hình học ba cạnh, giữ checkpoint, mặt nạ, và nguồn đề xuất F205 cố định. Chấm thử hình học bằng cùng đối sánh. Mốc sàng lọc: ít nhất 30 object Viewer mới và ít nhất 50% ứng viên bổ sung có đối sánh mới. Nếu đạt thì mới tích hợp vào hậu kỳ thực nghiệm và chấm F2/P/R object. Không train thêm hay đổi app theo số oracle này.

Kết quả H148. Ứng viên mới cho GT5 1/6 và Viewer8 17/32 theo phép ghép hình học. Cả 17 đối sánh Viewer thêm đều ở vw_f4afa52daee9_p0004. H148 không đạt mốc số lượng. Không đưa thuật toán vào app.

Kiểm tra bổ sung trên hai bộ object logic + thin đã cache, trước bước vùng kín. Chấm theo lớp đã quy đổi và IoU diện tích ≥ 0,50. Đổi từ 0,30 về 0,05 cho logic cho GT5 F2/P/R từ 0,880/0,898/0,876 thành 0,894/0,840/0,908; Viewer8 từ 0,200/0,595/0,171 thành 0,205/0,588/0,177. Kết quả này chỉ mô tả nhánh logic. User chuyển trọng tâm về kiến trúc bắt nét.
