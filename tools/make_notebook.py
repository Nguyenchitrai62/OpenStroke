"""Write colab/OpenStroke_train_quick.ipynb (run: py -3.12 tools/make_notebook.py)."""

import json
from pathlib import Path

CELLS = [
    ("md", """# OpenStroke: train trên Colab

Chạy lần lượt từng ô. GPU: *Runtime → Change runtime type* → **G4** (High-RAM).

## 1. Mount Google Drive"""),
    ("code", """from google.colab import drive
drive.mount('/content/drive')"""),
    ("md", """## 2. Cài OpenStroke

Luôn lấy bản mới nhất trên GitHub."""),
    ("code", """!pip install -q --force-reinstall --no-deps git+https://github.com/Nguyenchitrai62/OpenStroke.git"""),
    ("md", """## 3. Giải nén dataset

- **`openstroke_e2_colab.zip`** → `/content/dataset_local/e2`: FIRE + HVAC + 195 trang PLUMBING, nhãn v2 (HVAC bỏ nét bộ môn khác
  nằm dưới ống gió, sổ cái F18). `folds.json` = 5 fold FIRE / HVAC, `folds_plumbing.json` = thêm fold 5 PLUMBING. **Dùng cái này.**
- `openstroke_e0_colab.zip` / `openstroke_e1_colab.zip`: bộ cũ (nhãn v1), chỉ để chạy lại run cũ."""),
    ("code", """!for z in e2 e1 e0; do f="/content/drive/MyDrive/OpenStroke/openstroke_${z}_colab.zip"; [ -f "$f" ] && unzip -q -o "$f" -d /content/dataset_local && echo "unzipped $z"; done
"""),
    ("md", """## 4. (Tuỳ chọn) Đo tốc độ và VRAM trước khi train

Dữ liệu giả, khoảng 1 phút. Giữ `batch=4 window=65536` (các run so với nhau ở đúng mức này)."""),
    ("code", "!openstroke bench model=openstroke2 batch=2,4,8 window=65536"),
    ("md", """## 5. Train (dataset e2)

Năm run, mỗi run một ô, **chạy nguyên ô, lần lượt A2 → PMN → PMNG → PM → P**; gửi zip ô 7 sau mỗi run.
Giả thuyết ghi trước: `research/HYPOTHESES.md` E4 (H12–H15).

- Mỗi run tạo thư mục mới `runs/train`, `train2`... 5 fold chạy lần lượt (trên G4, `parallel=5` chậm hơn ~20 % tổng: sổ cái F21).
- Mất phiên: chạy lại ô 1–3, rồi `!openstroke train resume=true` (fold xong được bỏ qua, fold dở chạy tiếp).
- Đầu mỗi fold, log in mốc "đoán tất cả là ống" của fold đó: so điểm với mốc của chính fold.

**A2**: như run A cũ, trên nhãn mới (mốc cho mọi run sau)."""),
    ("code", """!openstroke train \\
data=/content/dataset_local/e2 \\
model=openstroke2 \\
fold=all \\
epochs=100 \\
restyle=0.5"""),
    ("md", """**PMN**: thêm 195 trang PLUMBING (fold 5 = cả dự án PLUMBING), model biết luồng (`mode_input`) và đọc tên layer
(`layer_names`). Đây là ứng viên chính để tăng độ chính xác (sổ cái F17, F19)."""),
    ("code", """!openstroke train \\
data=/content/dataset_local/e2 \\
folds=/content/dataset_local/e2/folds_plumbing.json \\
model=openstroke2 \\
fold=all \\
epochs=100 \\
restyle=0.5 \\
plumbing_share=0.3 \\
mode_input=true \\
layer_names=true"""),
    ("md", """**PMNG**: như PMN, thêm attention giữa các nhóm layer / nhóm style kiểu `layer_1` (`group_context=true`, H16)."""),
    ("code", """!openstroke train \\
data=/content/dataset_local/e2 \\
folds=/content/dataset_local/e2/folds_plumbing.json \\
model=openstroke2 \\
fold=all \\
epochs=100 \\
restyle=0.5 \\
plumbing_share=0.3 \\
mode_input=true \\
layer_names=true \\
group_context=true"""),
    ("md", """**PM**: như PMN, không tên layer (tách tác dụng của tên layer, H15)."""),
    ("code", """!openstroke train \\
data=/content/dataset_local/e2 \\
folds=/content/dataset_local/e2/folds_plumbing.json \\
model=openstroke2 \\
fold=all \\
epochs=100 \\
restyle=0.5 \\
plumbing_share=0.3 \\
mode_input=true"""),
    ("md", """**P**: như PM, không mode (tách tác dụng của dữ liệu PLUMBING, H12)."""),
    ("code", """!openstroke train \\
data=/content/dataset_local/e2 \\
folds=/content/dataset_local/e2/folds_plumbing.json \\
model=openstroke2 \\
fold=all \\
epochs=100 \\
restyle=0.5 \\
plumbing_share=0.3"""),
    ("md", """## 5c. (Cũ) Run trên e0, nhãn v1: A / C / B của E3

Chỉ để chạy lại run cũ, không so được với run trên e2."""),
    ("code", "!openstroke train \\\ndata=/content/dataset_local/e0 \\\nmodel=openstroke2 \\\nfold=all \\\nepochs=100 \\\nbatch=4 \\\nwindow=65536 \\\nlr0=0.002 \\\nrestyle=0.5 \\\nparallel=5\n\n# parallel=5                     (5 fold cùng lúc trên một GPU, mỗi fold ~17 GB; kết quả y như chạy lần lượt. OOM -> parallel=4)\n# restyle=0                     (hạt giống C: tắt augment đổi style, mọi thứ khác như A)\n# model=/content/runs/train      (fine-tune: mỗi fold bắt đầu từ fold<k>/best.pt của run đó)\n# resume=true                    (chạy tiếp run mới nhất sau khi mất phiên)\n# window=16384 batch=16          (hạt giống B của docs/OPENSTROKE2.md: chỉ đổi cửa sổ)\n# model=openstroke1 window=4096 batch=8 lr0=0.001 epochs=30   (v1 làm mốc)"),
    ("md", """## 6. (Một lần) Baseline cây GBDT, chạy CPU

Mốc H1: model phải vượt cây dùng cùng đặc trưng. Chạy một lần, khoảng 10–30 phút, không cần GPU."""),
    ("code", "!openstroke baseline-gbdt --data /content/dataset_local/e0 --folds /content/dataset_local/e0/folds.json --out runs/e1_gbdt"),
    ("md", """## 7. Tải log + `best.pt` của run mới nhất về máy

Gom `train.log`, `summary.json`, `best.pt` của mọi fold thành một zip. Gửi lại file zip này."""),
    ("code", """import os
import glob
import shutil
from google.colab import files

train_dirs = [d for d in glob.glob('runs/train*') if os.path.exists(os.path.join(d, 'train.log'))]

if train_dirs:
    latest_train_dir = max(train_dirs, key=os.path.getmtime)
    print(f"Thư mục train mới nhất được phát hiện: {latest_train_dir}")
    import json
    a = json.load(open(os.path.join(latest_train_dir, 'args.json')))
    folds_used = a.get('folds') or os.path.join(a['data'], 'folds.json')
    shutil.copy2(folds_used, os.path.join(latest_train_dir, 'folds_used.json'))
    tmp = os.path.join('/content/_pack', os.path.basename(latest_train_dir))
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(latest_train_dir, tmp, ignore=shutil.ignore_patterns('last.pt', 'val_scores.npz'))
    if os.path.isdir('runs/e1_gbdt'):
        shutil.copytree('runs/e1_gbdt', os.path.join(tmp, 'e1_gbdt'), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('*.npz', '*.pkl', '*.joblib'))
    zip_path = shutil.make_archive(f'/content/openstroke_{os.path.basename(latest_train_dir)}', 'zip', tmp)
    files.download(zip_path)
    print(f"Đang chuẩn bị tải file: {zip_path} ...")
else:
    print("Không tìm thấy bất kỳ thư mục train nào trong runs/")"""),
]

def main() -> None:
    nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"gpuType": "G4", "machine_shape": "hm", "provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    for kind, src in CELLS:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    out = Path(__file__).resolve().parents[1] / "colab" / "OpenStroke_train_quick.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
