# Colab v15 / E99: `colab/OpenStroke15_E99.ipynb`

VM GPU mới, chạy từ trên xuống: mount, cài `main`, giải nén ZIP cũ, rồi hai run:

```python
!openstroke obb train   data=/content/openstroke9_e83/obb_hvac7/data.yaml   model=/content/drive/MyDrive/Train/OpenStroke10/e85_graph2/weights/best.pt   instances=True   epochs=10   project=/content/drive/MyDrive/Train/OpenStroke15   name=e99_instances
```

Đối chứng: cùng lệnh với `instances=False` và `name=e99c_control`.

Mặc định của checkpoint OpenOBB25 (`openstroke.obb.cli.vector_defaults`, chỉ ghi khi muốn đổi): ô 1024 px tỉ lệ 1, 8.192 token,
batch 32 (nominal 64), lr0 1e-4, warmup 0,5 epoch, best.pt chọn theo F2 object. `instances=True` = 256 truy vấn (128 tự do) + ngữ cảnh
nét; `instances=False` = đối chứng; cả hai tắt nhánh đồ thị kế thừa từ E85. Data HVAC7 có trang pipeline được tự lọc một lần còn 18 / 8
trang GT vào `<thư mục>_gt` (`gt_only=False` để tắt). Bỏ `pilot_epochs`: run ngắn thì đặt `epochs=` nhỏ.
