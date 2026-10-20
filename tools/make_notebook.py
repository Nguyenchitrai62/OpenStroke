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

Zip giải nén ra `/content/dataset_local/e0` (các file `.npz`, `manifest.json`, `folds.json`).
Đã chốt fold riêng thì chép file của bạn đè lên `/content/dataset_local/e0/folds.json`."""),
    ("code", """!unzip -q -o "/content/drive/MyDrive/OpenStroke/openstroke_e0_colab.zip" -d /content/dataset_local
"""),
    ("md", """## 4. Train

- Cross-validation theo PDF: `--fold all` là cả 5 fold, hoặc `--fold 0,2`.
- Mỗi lần train tạo một thư mục mới: `runs/train`, `train2`... Trong đó `train.log`, `summary.json`, `fold<k>/best.pt`.
- Thêm `--no-layer-pool` để chạy bản so sánh không LayerPool.
- Lần đầu tính đặc trưng mất vài phút (cache ở `runs/cache`, các lần sau dùng lại).
- Mất phiên giữa chừng: chạy lại ô 1–3 rồi train các fold còn thiếu, ví dụ `--fold 3,4`."""),
    ("code", """!openstroke train \\
--data /content/dataset_local/e0 \\
--folds /content/dataset_local/e0/folds.json \\
--fold all \\
--epochs 30 \\
--batch 8 \\
--lr 0.001
"""),
    ("md", """## 5. Tải log + `best.pt` của run mới nhất về máy

Gom `train.log`, `summary.json`, `best.pt` của mọi fold thành một zip. Gửi lại file zip này."""),
    ("code", """import os
import glob
import shutil
from google.colab import files

train_dirs = [d for d in glob.glob('runs/train*') if os.path.exists(os.path.join(d, 'train.log'))]

if train_dirs:
    latest_train_dir = max(train_dirs, key=os.path.getmtime)
    print(f"Thư mục train mới nhất được phát hiện: {latest_train_dir}")
    shutil.copy2('/content/dataset_local/e0/folds.json', os.path.join(latest_train_dir, 'folds_used.json'))
    tmp = os.path.join('/content/_pack', os.path.basename(latest_train_dir))
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(latest_train_dir, tmp, ignore=shutil.ignore_patterns('last.pt', 'val_scores.npz'))
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
