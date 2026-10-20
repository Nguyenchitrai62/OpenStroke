"""Write colab/OpenStroke_train.ipynb (run: py -3.12 tools/make_notebook.py)."""

import json
from pathlib import Path

CELLS = [
    ("md", """# OpenStroke: train trên Colab (E1 baseline + E2 model v1)

Notebook: mount Drive → giải nén `openstroke_e0_colab.zip` (code + data E0) → baseline E1 → model v1 (cross-validation theo PDF) → gom log.

- Chỉ cần sửa ô **2** (đường dẫn zip trên Drive) và nếu muốn, ô **6** (siêu tham số).
- GPU: *Runtime → Change runtime type* → **G4** (High-RAM) hoặc T4. Model nhỏ (~1,1M tham số), dữ liệu ~118 MB.
- **Fold:** zip mang `folds.json` = bản *đề xuất* (`status: proposal`). Mọi log in dòng `WARNING: fold file is not confirmed`
  cho tới khi bạn chốt. Muốn sửa fold hoặc chọn bản GT cho 5 trang trùng: sửa file JSON, chép lên Drive, đặt `FOLDS_OVERRIDE` ở ô 2.
- Gửi lại: file `openstroke_logs_<giờ>.zip` (ô 8 chép lên Drive), hoặc dán các dòng `SUMMARY`, `GBDT SUMMARY`, `ALL:` và `PAGE`."""),
    ("md", "## 1. Mount Google Drive"),
    ("code", """from google.colab import drive
drive.mount('/content/drive')"""),
    ("md", "## 2. Đường dẫn"),
    ("code", """ZIP = "/content/drive/MyDrive/OpenStroke/openstroke_e0_colab.zip"   # file bạn đẩy lên Drive
FOLDS_OVERRIDE = None    # ví dụ "/content/drive/MyDrive/OpenStroke/folds.json" sau khi bạn chốt fold
DRIVE_OUT = "/content/drive/MyDrive/OpenStroke/runs"                 # nơi chép log + weights về
WORK = "/content/openstroke"
"""),
    ("md", "## 3. Giải nén và cài OpenStroke (từ zip, chưa có remote Git)"),
    ("code", """import os, shutil, json
if not os.path.exists(WORK + "/.unzipped"):
    !unzip -q -o "{ZIP}" -d "{WORK}"
    open(WORK + "/.unzipped", "w").close()
if FOLDS_OVERRIDE:
    shutil.copy2(FOLDS_OVERRIDE, WORK + "/e0/folds.json")
!pip install -q --no-deps -e "{WORK}/openstroke_src"
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import openstroke, torch, sklearn, scipy, shapely
print("OpenStroke", openstroke.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
folds = json.load(open(WORK + "/e0/folds.json", encoding="utf-8"))
print("fold status:", folds["status"], "| k =", folds["k"], "| duplicate_choice:", folds.get("duplicate_choice"))
for g in folds["groups"]:
    print(f"  fold {g['fold']}  {g['mode']:4s}  pages {g['pages']}  {g['filename']}")"""),
    ("md", """## 4. E1.1 Trần của cách chọn theo layer (dùng nhãn GT, chỉ numpy, vài giây)

Với mỗi trang, tập layer tốt nhất theo F2 độ dài. Trang không có layer thật (như `7. FP_page 16`) không cứu được bằng layer."""),
    ("code", """%cd {WORK}
!python -m openstroke baseline-layer --data e0 --folds e0/folds.json --out runs/e1_layer_ceiling.json"""),
    ("md", """## 5. E1.3 Baseline cây (HistGradientBoosting trên đặc trưng tay, CPU, khoảng 5–15 phút)

Cùng fold, cùng đặc trưng (`openstroke/features.py`, 47 cột) và cùng metric với model v1. Nếu cây đã gần bằng model thì
phần kiến trúc chưa đáng tiền (AGENTS.md mục 5). Lần chạy đầu tạo cache đặc trưng (`runs/cache`), model dùng lại."""),
    ("code", """!python -m openstroke baseline-gbdt --data e0 --folds e0/folds.json --out runs/e1_gbdt --cache runs/cache"""),
    ("md", """## 6. E2 Model v1: hai hạt giống khác đúng một yếu tố (`docs/OPENSTROKE1.md`)

- **A** `v1`: kNN attention + LayerPool (mặc định).
- **B** `v1_nolp`: y hệt, tắt LayerPool (`--no-layer-pool`). So A với B đo đúng phần đóng góp của LayerPool.

Mỗi lệnh chạy 5 fold liên tiếp. Mất phiên giữa chừng: chạy lại ô 1–3 và chỉ fold còn thiếu, ví dụ `--fold 3,4`."""),
    ("code", """EPOCHS, STEPS, BATCH, WINDOW = 30, 200, 8, 4096
COMMON = f"--data e0 --folds e0/folds.json --cache runs/cache --epochs {EPOCHS} --steps {STEPS} --batch {BATCH} --window {WINDOW}"
!python -m openstroke train {COMMON} --out runs/v1"""),
    ("code", """!python -m openstroke train {COMMON} --out runs/v1_nolp --no-layer-pool"""),
    ("md", "## 7. Tóm tắt nhanh"),
    ("code", """import glob
for f in ["runs/e1_layer_ceiling.json", "runs/e1_gbdt/gbdt_summary.json", "runs/v1/summary.json", "runs/v1_nolp/summary.json"]:
    if not os.path.exists(f):
        print("thiếu", f); continue
    d = json.load(open(f))
    key = "all" if "all" in d else ("all_at_fold_threshold" if "all_at_fold_threshold" in d else "at_fold_threshold")
    print(f, json.dumps(d[key], indent=None)[:400])"""),
    ("md", "## 8. Gom log + weights, chép lên Drive (gửi file zip này lại)"),
    ("code", """import time
stamp = time.strftime("%m%d_%H%M")
out = f"/content/openstroke_logs_{stamp}"
os.makedirs(out, exist_ok=True)
for f in glob.glob("runs/**/*", recursive=True):
    if "cache" in f or os.path.isdir(f):
        continue
    if f.endswith((".log", ".json")) or f.endswith("best.pt"):
        dst = os.path.join(out, os.path.relpath(f, "runs"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(f, dst)
shutil.copy2("e0/folds.json", out + "/folds_used.json")
shutil.make_archive(out, "zip", out)
os.makedirs(DRIVE_OUT, exist_ok=True)
shutil.copy2(out + ".zip", DRIVE_OUT)
print("đã chép", out + ".zip", "->", DRIVE_OUT)"""),
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
