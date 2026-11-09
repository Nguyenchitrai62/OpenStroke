"""Run OpenOBB screens as Kaggle kernels (the user's account; token from the repo's .env, never printed).

    py -3.12 tools/kaggle_run.py dataset <slug> <folder>          # create or version a private dataset from a folder
    py -3.12 tools/kaggle_run.py push <name> '<json cfg>'           # a GPU kernel from tools/kaggle_obb_kernel.py with that config
    py -3.12 tools/kaggle_run.py status <name> [<name> ...]
    py -3.12 tools/kaggle_run.py fetch <name> [<name> ...]          # outputs -> runs_kaggle/<name>/

Config keys: name, ckpt (file name inside a dataset), subset ("base" / "base+rs" / "base+cp"), epochs, extra (train args), hours,
datasets (list of "<user>/<slug>").
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env():
    tok = (ROOT / ".env").read_text(encoding="utf-8").split("=", 1)[1].strip()
    return dict(os.environ, KAGGLE_API_TOKEN=tok, PYTHONIOENCODING="utf-8")


def kaggle(*args, check=True):
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, env=env(), encoding="utf-8", errors="replace")
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "outdated" not in l)
    if check and r.returncode:
        raise SystemExit(out)
    return out


def user():
    return "nguynchtrai"


def dataset(slug, folder):
    folder = Path(folder)
    meta = folder / "dataset-metadata.json"
    if not meta.exists():
        meta.write_text(json.dumps({"title": slug, "id": f"{user()}/{slug}", "licenses": [{"name": "other"}]}), encoding="utf-8")
    existing = kaggle("datasets", "status", f"{user()}/{slug}", check=False)
    if "ready" in existing.lower() or "pending" in existing.lower():
        print(kaggle("datasets", "version", "-p", str(folder), "-m", "update", "--dir-mode", "zip"))
    else:
        print(kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "zip"))


def push(name, cfg_json):
    cfg = json.loads(cfg_json)
    cfg["name"] = name
    d = ROOT / "runs_kaggle" / name / "kernel"
    d.mkdir(parents=True, exist_ok=True)
    body = (ROOT / "tools" / "kaggle_obb_kernel.py").read_text(encoding="utf-8").replace("__CFG__", json.dumps(cfg))
    (d / "run.py").write_text(body, encoding="utf-8")
    meta = {"id": f"{user()}/openstroke-{name}", "title": f"openstroke-{name}", "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": cfg["datasets"], "competition_sources": [], "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    acc = cfg.get("accelerator")
    print(kaggle("kernels", "push", "-p", str(d), *(["--accelerator", acc] if acc else [])))


def status(*names):
    for n in names:
        print(n, kaggle("kernels", "status", f"{user()}/openstroke-{n}", check=False).strip().splitlines()[-1])


def fetch(*names):
    for n in names:
        d = ROOT / "runs_kaggle" / n
        d.mkdir(parents=True, exist_ok=True)
        print(kaggle("kernels", "output", f"{user()}/openstroke-{n}", "-p", str(d), "-o", check=False)[-600:])


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"dataset": dataset, "push": push, "status": status, "fetch": fetch}[cmd](*rest)
