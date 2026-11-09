"""Kaggle kernel body for an OpenOBB screen (filled in by ``tools/kaggle_run.py``; runs on Kaggle, never locally).

Installs OpenStroke from the code dataset, builds the experiment's own dataset from the shared data zip (train subset chosen by file-name
suffix: base pages, + ``_rs*`` restyled copies, + ``_cp`` copy-paste pages; val always the same 8 pages), trains (fine-tune from the given
checkpoint), then writes the val detections for ``research/probes/obb_score_probe.py``. Everything the run produces is under
/kaggle/working (downloaded with ``kaggle kernels output``).
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

CFG = json.loads(r'''__CFG__''')
WORK = Path("/kaggle/working")
EXP = Path("/tmp/exp")
log = open(WORK / "kernel.log", "a", buffering=1)


def say(*a):
    print(*a, flush=True)
    print(*a, file=log, flush=True)


def sh(cmd, **kw):
    say("$", cmd)
    return subprocess.run(cmd, shell=True, **kw)


t0 = time.time()
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
tars = glob.glob("/kaggle/input/**/openstroke.tar.gz", recursive=True)
code = Path(tars[0]) if tars else Path(sorted(glob.glob("/kaggle/input/**/openstroke/pyproject.toml", recursive=True))[0]).parent  # Kaggle unpacks archives
# no pip install (a quiet install landed in another interpreter, A1 v1): the package is imported from the code folder directly
os.environ["PYTHONPATH"] = f"{code}:" + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(code))
sh("pip install -q shapely opencv-python-headless pyyaml 2>&1 | tail -1")
ckpt = glob.glob(f"/kaggle/input/**/{CFG['ckpt']}", recursive=True)[0]
src = glob.glob("/kaggle/input/**/obb_hvac9/data.yaml", recursive=True)
if src:
    src = Path(src[0]).parent
else:  # the zip was kept as a zip
    z = glob.glob("/kaggle/input/**/openstroke_obb_hvac9.zip", recursive=True)[0]
    zipfile.ZipFile(z).extractall("/tmp/data")
    src = Path("/tmp/data/obb_hvac9")
say("data", src, "ckpt", ckpt)


def wanted(stem: str) -> bool:
    if "_rs" in stem:
        return "rs" in CFG["subset"]
    if stem.endswith("_cp"):
        return "cp" in CFG["subset"]
    return True


for split in ("train", "val"):
    (EXP / "images" / split).mkdir(parents=True, exist_ok=True)
    (EXP / "labels" / split).mkdir(parents=True, exist_ok=True)
    for img in (src / "images" / split).glob("*.png"):
        if split == "val" or wanted(img.stem):
            os.symlink(img, EXP / "images" / split / img.name)
            os.symlink(src / "labels" / split / f"{img.stem}.txt", EXP / "labels" / split / f"{img.stem}.txt")
tools = code / "tools" if (code / "tools").exists() else Path(glob.glob("/kaggle/input/**/tools/synth_hvac.py", recursive=True)[0]).parent
if "cp" in CFG["subset"]:  # rare objects pasted into blank paper of other train pages (done here: needs RAM the local machine lacks)
    sh(f"python {tools}/obb_copy_paste.py {EXP} --copies 40")
if CFG.get("synth"):  # E78: procedural HVAC pages with exact labels, mixed into train
    sh(f"python {tools}/synth_hvac.py {EXP} --pages {CFG['synth']} --size 900x650 --scale 4 --seed 1")
yaml = (src / "data.yaml").read_text(encoding="utf-8").replace("path: .", f"path: {EXP}")
(EXP / "data.yaml").write_text(yaml, encoding="utf-8")
say("train images", len(list((EXP / "images" / "train").glob("*.png"))), "val images", len(list((EXP / "images" / "val").glob("*.png"))))

name = CFG["name"]
cmd = (f"python -m openstroke obb train data={EXP}/data.yaml model={ckpt} epochs={CFG['epochs']} imgsz=1024 tile=True tile_scale=1 "
       f"{CFG.get('extra', '')} project={WORK}/runs name={name} exist_ok=True")
import re

VAL = re.compile(r"^\s+all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
floor, patience = CFG.get("stop_below"), CFG.get("stop_patience", 2)  # early stop: val mAP50 under the floor for N evals in a row
say("$", cmd)
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
curve, below, rc = [], 0, None
with open(WORK / "train.log", "w") as fh:
    for line in proc.stdout:
        fh.write(line)
        for part in line.replace(chr(13), chr(10)).split(chr(10)):  # tqdm redraws with carriage returns
            m = VAL.match(part)
            if m:
                curve.append(float(m.group(3)))
                below = below + 1 if floor is not None and curve[-1] < floor else 0
                say(f"val {len(curve)}: mAP50 {curve[-1]:.3f}" + (f" (below floor {floor} x{below})" if below else ""))
                if floor is not None and below >= patience:
                    say("EARLY STOP: val mAP50 stays under", floor)
                    proc.kill()
                    rc = "early stop"
        if time.time() - t0 > CFG.get("hours", 8) * 3600:
            proc.kill()
            rc = "timeout"
    proc.wait()
rc = rc if rc is not None else proc.returncode
(WORK / "curve.json").write_text(json.dumps({"map50": curve, "exit": str(rc)}), encoding="utf-8")
say("train exit", rc, f"{(time.time() - t0) / 3600:.2f} h")
best = WORK / "runs" / name / "weights" / "best.pt"
if best.exists():
    from openstroke.obb.cli import predict

    dets = predict(str(best), str(EXP / "images" / "val"), conf=0.05, tile_scale=1, save=False)
    (WORK / "val_dets.json").write_text(json.dumps({Path(k).stem: v for k, v in dets.items()}, default=float), encoding="utf-8")
    say("val dets", {Path(k).stem: len(v) for k, v in dets.items()})
for f in glob.glob(str(WORK / "runs" / name / "weights" / "last.pt")):
    os.remove(f)  # keep the output small: best.pt is enough
shutil.rmtree(WORK / "runs" / name / "cache", ignore_errors=True)
say("done", f"{(time.time() - t0) / 3600:.2f} h")
