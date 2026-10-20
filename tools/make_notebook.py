"""Write colab/OpenStroke_train_quick.ipynb (run: py -3.12 tools/make_notebook.py)."""

import json
from pathlib import Path

CELLS = [
    ("md", """# OpenStroke: train nhanh trên Colab (CLI)

Notebook gọn dùng hằng ngày: mount Drive → cài OpenStroke → giải nén dataset → `openstroke train` → lấy log + `best.pt`.

- Chỉ cần sửa ô **3** (chọn dataset) và ô **4** (epoch, batch...).
- Mỗi lần train tạo thư mục mới `runs/train`, `train2`... Trong đó: `train.log`, `summary.json`, `fold<k>/best.pt`.
- Mất phiên giữa chừng: chạy lại ô 1–3 rồi train lại các fold còn thiếu (`FOLD = "3,4"`).
- GPU: *Runtime → Change runtime type* → **G4** (High-RAM)."""),
    ("md", "## 1. Mount Google Drive"),
    ("code", """from google.colab import drive
drive.mount('/content/drive')"""),
    ("md", "## 2. Cài OpenStroke (bản mới nhất trên GitHub)"),
    ("code", """!pip install -q --force-reinstall --no-deps git+https://github.com/Nguyenchitrai62/OpenStroke.git
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import openstroke; print("OpenStroke", openstroke.__version__)"""),
    ("md", """## 3. Giải nén dataset

Đổi `DATASET` để chọn. `folds.json` trong zip là bản đề xuất; đã chốt fold riêng thì đặt `FOLDS` là file của bạn trên Drive."""),
    ("code", """import glob, os, shutil

ZIPS = {
    "e0": "/content/drive/MyDrive/OpenStroke/openstroke_e0_colab.zip",
}
DATASET = "e0"
FOLDS   = None   # ví dụ "/content/drive/MyDrive/OpenStroke/folds.json"

dst = f"/content/dataset_local/{DATASET}"
if not os.path.exists(dst + "/.unzipped"):            # chỉ giải nén một lần mỗi phiên
    !unzip -q -o "{ZIPS[DATASET]}" -d "{dst}"
    open(dst + "/.unzipped", "w").close()

DATA = f"{dst}/e0"
if FOLDS:
    shutil.copy2(FOLDS, f"{DATA}/folds.json")
print(DATA, "-", len(glob.glob(f"{DATA}/*.npz")), "trang")"""),
    ("md", """## 4. Train

- Chạy cross-validation theo PDF: `FOLD = "all"` là cả 5 fold, hoặc `"0,2"`.
- `EXTRA = "--no-layer-pool"` để chạy bản so sánh không LayerPool."""),
    ("code", """FOLD   = "all"
EPOCHS = 30
BATCH  = 8
LR     = 0.001
EXTRA  = ""

n = len(glob.glob("/content/runs/train*"))
RUN = "/content/runs/train" + (str(n + 1) if n else "")
!openstroke train --data {DATA} --folds {DATA}/folds.json --cache /content/runs/cache --out {RUN} --fold {FOLD} --epochs {EPOCHS} --batch {BATCH} --lr {LR} {EXTRA}"""),
    ("md", """## 5. Lấy log + `best.pt` của run mới nhất

Gom `train.log`, `summary.json` và `best.pt` của mọi fold thành một zip; tải về máy, và chép lên Drive nếu đặt `DRIVE_OUT`.
Gửi lại file zip này."""),
    ("code", """from google.colab import files

DRIVE_OUT = None   # ví dụ "/content/drive/MyDrive/OpenStroke/runs"

runs = [r for r in glob.glob("/content/runs/train*") if os.path.exists(r + "/train.log")]
if not runs:
    print("Không tìm thấy run nào trong /content/runs/")
else:
    run = max(runs, key=os.path.getmtime)
    shutil.copy2(f"{DATA}/folds.json", run + "/folds_used.json")
    tmp = "/content/_pack/" + os.path.basename(run)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(run, tmp, ignore=shutil.ignore_patterns("last.pt", "val_scores.npz"))
    out = shutil.make_archive("/content/openstroke_" + os.path.basename(run), "zip", tmp)
    print("Run mới nhất:", run, "->", out)
    if DRIVE_OUT:
        os.makedirs(DRIVE_OUT, exist_ok=True)
        shutil.copy2(out, DRIVE_OUT)
        print("Đã chép lên Drive:", DRIVE_OUT)
    files.download(out)"""),
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
