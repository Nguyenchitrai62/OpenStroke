"""Write colab/OpenStroke_train.ipynb (run: py -3.12 tools/make_notebook.py)."""

import json
from pathlib import Path

CELLS = [
    ("md", """# OpenStroke: train trên Colab

Notebook gọn: mount Drive → cài OpenStroke → giải nén data → `openstroke train` → gom log.

- Chỉ cần sửa ô **3** (đường dẫn zip) và ô **4** (epoch, batch...).
- Kết quả ở `runs/<tên>/`: `train.log`, `summary.json`, `fold<k>/best.pt`.
- Mất phiên giữa chừng: chạy lại ô 1–3 rồi train tiếp các fold còn thiếu, ví dụ `FOLD = "3,4"`.
- GPU: *Runtime → Change runtime type* → **G4** hoặc T4."""),
    ("md", "## 1. Mount Google Drive"),
    ("code", """from google.colab import drive
drive.mount('/content/drive')"""),
    ("md", "## 2. Cài OpenStroke (bản mới nhất trên GitHub)"),
    ("code", """!pip install -q --force-reinstall --no-deps git+https://github.com/Nguyenchitrai62/OpenStroke.git
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import openstroke; print("OpenStroke", openstroke.__version__)"""),
    ("md", """## 3. Giải nén data

`folds.json` trong zip là bản đề xuất. Đã chốt fold riêng thì đặt `FOLDS` là file của bạn trên Drive."""),
    ("code", """import os, shutil
ZIP   = "/content/drive/MyDrive/OpenStroke/openstroke_e0_colab.zip"
FOLDS = None    # ví dụ "/content/drive/MyDrive/OpenStroke/folds.json"

if not os.path.exists("/content/e0/.unzipped"):     # chỉ giải nén một lần mỗi phiên
    !unzip -q -o "{ZIP}" -d /content
    open("/content/e0/.unzipped", "w").close()
if FOLDS:
    shutil.copy2(FOLDS, "/content/e0/folds.json")
%cd /content"""),
    ("md", """## 4. Train

- `NAME`: tên run (thư mục `runs/NAME`). `FOLD = "all"` chạy hết 5 fold.
- Thêm `--no-layer-pool` vào `EXTRA` để chạy bản so sánh không LayerPool (đổi `NAME`)."""),
    ("code", """NAME   = "v1"
FOLD   = "all"
EPOCHS = 30
BATCH  = 8
EXTRA  = ""      # ví dụ "--no-layer-pool"

!python -m openstroke train --data e0 --folds e0/folds.json --cache runs/cache --out runs/{NAME} --fold {FOLD} --epochs {EPOCHS} --batch {BATCH} {EXTRA}"""),
    ("md", "## 5. Chép log và weights lên Drive (gửi lại file zip này)"),
    ("code", """import time
DRIVE_OUT = "/content/drive/MyDrive/OpenStroke/runs"
out = f"/content/openstroke_{NAME}_{time.strftime('%m%d_%H%M')}"
shutil.copytree(f"runs/{NAME}", out, ignore=shutil.ignore_patterns("cache", "last.pt", "val_scores.npz"))
shutil.copy2("e0/folds.json", out + "/folds_used.json")
shutil.make_archive(out, "zip", out)
os.makedirs(DRIVE_OUT, exist_ok=True)
shutil.copy2(out + ".zip", DRIVE_OUT)
print("Đã chép lên Drive:", os.path.join(DRIVE_OUT, os.path.basename(out) + ".zip"))"""),
    ("md", """## 6. (Tuỳ chọn) Baseline cây để so với model

Cùng fold, cùng đặc trưng; chạy CPU khoảng 5–15 phút. Gửi lại dòng `GBDT SUMMARY`."""),
    ("code", """!python -m openstroke baseline-gbdt --data e0 --folds e0/folds.json --cache runs/cache --out runs/gbdt"""),
]

def main() -> None:
    nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    for kind, src in CELLS:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    out = Path(__file__).resolve().parents[1] / "colab" / "OpenStroke_train.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
