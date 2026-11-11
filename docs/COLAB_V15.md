# Colab v15 — pilot10 rồi quyết định train tiếp

Giữ ômountDrive/unzipZIPcũ. Khônguploaddatasetmới. V15lọcGT18train/8val,loại25pseudo;khôngscore_pack.
E69pathdướiđâylàquyướcnotebookcũ:nếubạnlưuE69ởnơikhác,chỉđổi model= chođúngfile. Không dùngv12graph đểthay thếngầminitialization.

## 1. Cài code mới

```python
!pip install -q --force-reinstall --no-deps "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
```

## 2. Lọc GT từ dữ liệu đã giải nén

```python
!python -m openstroke.obb.gt_dataset \
  --source /content/openstroke9_e83/obb_hvac7 \
  --out /content/openstroke15_gt
```

## 3. Pilot10 với lịchLR50

```python
!openstroke obb train \
  data=/content/openstroke15_gt/data.yaml \
  model=/content/drive/MyDrive/Train/OpenStroke9/e69/best.pt \
  epochs=50 \
  pilot_epochs=10 \
  batch=32 \
  nbs=32 \
  lr0=0.0001 \
  warmup_epochs=0.5 \
  seed=0 \
  imgsz=1024 \
  tile=True \
  tile_scale=1 \
  max_tokens=8192 \
  max_iters=190 \
  mosaic=0 \
  scale=0.25 \
  instance_queries=256 \
  instance_free=128 \
  instance_line_context=True \
  instance_f2=False \
  stroke_graph=False \
  patience=0 \
  project=/content/drive/MyDrive/Train/OpenStroke15 \
  name=e99_instances_gt
```

`epochs=50 pilot_epochs=10` tựdừngsau10epoch,giữnguyênlịchLR/EMA/optimizer của50epoch. Khôngphảichạy50epochngay.
Chạy20epochpilotthìđổi pilot_epochs=20. Khoảng52batch/epochởGT-only,báologthực tế làchuẩn. F2/R/P object làchỉsốchính, best.pt làcheckpointdeploy.

## 4. Sau khi xem log: tiếp tới50 hoặc tiếp pilot20

Không chạycellnàytrướckhiđánhgiápilot. Tiếprunđúngoptimizer/LR:

```python
!openstroke obb train \
  resume=/content/drive/MyDrive/Train/OpenStroke15/e99_instances_gt/weights/last.pt \
  pilot_epochs=0
```

Dừnglạiở20đểtesttiếp:thay pilot_epochs=0 bằng20. Dùng last.pt đểresume,best.pt đểchạyapp. NếuCLIđãtựtạotênruncóhậutố2/3,đổiđườngresume theo save_dirtronglog.

## 5. Đối chứng và ablation

Cùngdata/init/recipe,đổiname riêng. E98c:instance_queries=0,bỏinstance_free/instance_line_context/instance_f2. E98:instance_free=0. E100:instance_f2=True. H115:instance_line_context=False. Không sođiểmGT-onlymới vớiE69mixed0,511 nhưcùngthínghiệm.

Mốc kỳvọng nghiêncứu:10ep0,60;20ep0,70;50ep0,80. Chưacóaccuracyv15; testkỹthuật khôngđảmbảođạtđiểm. Nhánhvector thamkhảoVecFormer/CADSpotting/SymPoint/PTv3 theo thẻ OPENSTROKE15.md.
