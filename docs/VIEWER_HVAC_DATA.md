# Dữ liệu Viewer HVAC đã rà lại — kế hoạch, chốt kiểm và đối chứng

User 2026-11-23 cho phép thêm trang Viewer HVAC vào train **sau khi dự án #19 được mở lại và trang được rà lại** (AGENTS 0.13). Thẻ này ghi điều
kiện, cách kiểm từng trang và thí nghiệm đối chứng **trước khi có dữ liệu**. Chưa dựng dataset, chưa train.

## 1. Vì sao

- Học thuộc: F2 nét trang train 0,993, trang GT lạ 0,933 (F182). Ở mức object, E128 / E134 có train3 tăng đều trong khi val dừng ở 0,66–0,68
  rồi giảm từ epoch 6–7 (F198, E134).
- Các cách chống học thuộc bằng augment / đầu vào / loss đều không vượt đối chứng trên trang lạ: trộn trang E123 / E124 / E126, bỏ đặc trưng trang
  E125, loss cặp khó E132, cân bằng PDF E129 (F184–F192, F200–F211).
- Nhánh GT-only chỉ có 19 trang train từ khoảng 10 PDF; doc214 chiếm khoảng 47 % cửa sổ trước cân bằng (F199). Đánh giá: nút thắt là độ đa dạng
  dữ liệu, không phải dung lượng model.
- Dữ liệu Viewer hiện có **chưa rà**: e12 đã có 171 trang train / 94 PDF + 8 trang val từ bộ "Labels V1" (0 trang `completed` ngày 2026-11-10).
  E114–E127 train trên chúng vẫn lệch như trên; nhãn không nhất quán trong một trang (F186), quy ước lớp khác GT (F162). Thêm trang cùng chất lượng
  dạy thêm mâu thuẫn: ống trông như ống mà không có nhãn chính là lỗi bắt thừa lớn nhất.

## 2. Điều kiện trước khi dựng (ngoài tầm agent)

1. Người quản lý mở lại dự án **#19 "HVAC ready — Labels V1"** trong "Vòng đời dữ liệu" (MCP trả `PROJECT_LOCAL_ARCHIVED`; tài khoản của agent
   không có quyền `manage_project`). Các dự án HVAC khác (lo1–lo5, data, machine, flex_duct) chỉ có ảnh, không dùng cho model vector.
2. Đội gán nhãn rà lại một nhóm trang và đánh dấu `completed`:
   - **Gán đủ mọi ống trên trang** (Duct, Elbow, Elbow vuông, Reducer, Chân rẽ, Tee); không bỏ đoạn nào vì "giống phần đã gán".
   - Đúng 6 lớp chung; ống kèm máy, Z, cross để riêng như hiện nay.
   - Ưu tiên **PDF / dự án chưa có trong train**, tối đa 3 trang mỗi PDF (đa dạng hơn số lượng). Khởi đầu đề xuất 30–50 trang.

## 3. Chốt kiểm từng trang (agent, tự động, trước khi vào train)

Dựng bằng đường sẵn có: `tools/viewer_obb_dataset.py extract | build` → `tools/make_stroke_dataset.py` → `tools/make_e11.py` / `make_e12.py`.
Các chốt sẵn có giữ nguyên: loại tài liệu cùng dự án với trang GT (`EXCLUDE`), trang chung > 30 % đầu mút nét dài với trang GT (`fingerprint`),
hộp không rơi lên nét (`calibrate`), quá nhiều object không có nét (`--max-bare`), `--max-pages-per-doc`. Thêm:

| Chốt | Quy tắc | Xử lý khi trượt |
|---|---|---|
| Trạng thái | chỉ `completed` của bộ nhãn object | bỏ trang |
| Lớp | mọi tên lớp ánh xạ được về 6 lớp chung | bỏ object lạ, báo danh sách |
| Độ phủ theo kiểu nét | với mỗi nhóm kiểu nét (màu + độ dày + loại) có nét thuộc object, phần chiều dài nhóm đó nằm ngoài mọi object | > 25 %: trang nghi thiếu nhãn, không đưa vào train, xuất danh sách vùng cho đội gán nhãn |
| Trùng GT | trang trùng (sha256, trang) với GT AI_Takeoff | GT thắng; đo độ khớp nhãn trên trang trùng để ước chất lượng |
| Provenance | manifest ghi nguồn `viewer19` / `aitakeoff_gt`, bộ nhãn, ngày xuất | — |

Chỉ dùng hình học và nhãn của chính trang; không dùng điểm model để lọc hay sửa nhãn (không nhãn giả). Ngưỡng 25 % là đề xuất, chốt lại khi đã
thấy phân bố trên trang rà xong, trước khi train.

## 4. Split và đối chứng

- Trang Viewer chỉ vào **train**. Val chính giữ nguyên **GT5** (doc180, 187, 189, 200, 201), cùng cohort với E128 / E134. PDF / dự án của GT5
  không được có trang Viewer trong train.
- Số phụ: giữ riêng cả một số dự án Viewer đã rà làm val Viewer (không trộn PDF giữa train / val) để đo trên quy ước gán nhãn Viewer.
- Thí nghiệm một biến: cùng kiến trúc, khởi tạo, lịch, seed; đối chứng GT-only (19 trang) so với GT + Viewer đã rà. Kiến trúc chọn khi viết thẻ
  version (nhánh bắt nét E114 hoặc OpenStroke35 E128). Tiêu chí đạt / bác bỏ và kỳ vọng định lượng ghi vào thẻ trước GPU.
- Chấm bằng F2 object in trong log (`openstroke/object_eval.py`), từng trang, từng lớp; báo cả khoảng train3 / val để thấy học thuộc giảm hay không.

## 5. Trạng thái

- 2026-11-23: user cho phép (AGENTS 0.13). Dự án #19 đang cất; chờ mở lại và rà trang. Chưa dựng dataset, chưa train.
