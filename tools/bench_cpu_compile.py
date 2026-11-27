"""CPU inference timing with ``torch.compile`` off vs on (one fresh process per setting).

    py -3.12 tools/bench_cpu_compile.py --model best_deploy.pt --pages small.json.gz large.json.gz [--repeat 3]

Each setting runs in its own subprocess (clean thread pool, no shared compile cache in memory) and reports, per page:
load time, first ``predict`` (includes compilation when on), steady-state mean of the next ``--repeat`` calls with the
preprocess / inference / postprocess split, and the max |Δprob| against compile off (parity check).

``torch.compile`` on CPU needs a C++ compiler on PATH (MSVC ``cl`` on Windows, from "Developer PowerShell for VS";
``g++`` / ``clang`` on Linux). Without one, the compiled setting is reported as unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a source checkout without pip install


def _force_cpu_compile(model, mode: str) -> None:
    """Compile the same ImageBranch submodules ``openstroke.infer._compile_submodules`` compiles on GPU."""
    import torch

    img = getattr(model, "img", None)
    if img is None:
        raise RuntimeError("checkpoint has no image branch: nothing to compile")
    for name in ("stem", "layer2", "layer3", "layer4", "cad_stem"):
        sub = getattr(img, name, None)
        if sub is not None:
            setattr(img, name, torch.compile(sub, mode=mode, dynamic=True))


def _child(args) -> None:
    import torch

    from openstroke.infer import OpenStroke, load_payload

    t0 = time.perf_counter()
    m = OpenStroke(args.model, device="cpu", threads=args.threads)
    if args.setting != "off":
        _force_cpu_compile(m.model, args.setting)
    load_s = time.perf_counter() - t0
    out = {"setting": args.setting, "threads": torch.get_num_threads(), "torch": torch.__version__,
           "load_s": load_s, "pages": []}
    for p in args.pages:
        payload = load_payload(p)
        t = time.perf_counter()
        r = m.predict(payload, verbose=False)[0]
        first = time.perf_counter() - t
        steady, parts = [], []
        for _ in range(args.repeat):
            t = time.perf_counter()
            r = m.predict(payload, verbose=False)[0]
            steady.append(time.perf_counter() - t)
            parts.append(r.speed)
        np.save(Path(args.out_dir) / f"{Path(p).name}.{args.setting}.npy", np.asarray(r.prob, np.float32))
        out["pages"].append({
            "page": Path(p).name, "shapes": len(payload["shapes"]), "first_s": first,
            "steady_s": float(np.mean(steady)), "steady_min_s": float(np.min(steady)),
            "speed_ms": {k: float(np.mean([s[k] for s in parts])) for k in parts[0]},
        })
    print("RESULT " + json.dumps(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pages", nargs="+", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--settings", default="off,default", help="comma list: off, default, max-autotune ...")
    ap.add_argument("--out-dir", default=None, help="where per-page prob arrays go (default: next to this run)")
    ap.add_argument("--setting", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = str(Path(os.environ.get("TEMP", ".")) / "openstroke_bench_cpu_compile")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.setting:
        _child(args)
        return

    results = {}
    for s in [x.strip() for x in args.settings.split(",") if x.strip()]:
        cmd = [sys.executable, __file__, "--model", args.model, "--pages", *args.pages, "--repeat", str(args.repeat),
               "--setting", s, "--out-dir", args.out_dir]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        print(f"== compile={s} ...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            err = (proc.stderr or proc.stdout).strip().splitlines()
            msg = next((ln for ln in reversed(err) if "Error" in ln or "error" in ln), err[-1] if err else "?")
            print(f"   unavailable: {msg[:300]}")
            results[s] = None
            continue
        results[s] = json.loads(line[len("RESULT "):])

    base = results.get("off")
    print()
    print(f"{'compile':<14}{'page':<26}{'shapes':>9}{'first s':>10}{'steady s':>10}{'pre ms':>9}{'infer ms':>10}"
          f"{'post ms':>9}{'speedup':>9}{'max|dp|':>10}")
    for s, res in results.items():
        if res is None:
            print(f"{s:<14}(unavailable)")
            continue
        for i, pg in enumerate(res["pages"]):
            sp = dp = ""
            if base is not None and s != "off":
                b = base["pages"][i]
                sp = f"{b['steady_s'] / pg['steady_s']:.2f}x"
                a0 = np.load(Path(args.out_dir) / f"{pg['page']}.off.npy")
                a1 = np.load(Path(args.out_dir) / f"{pg['page']}.{s}.npy")
                dp = f"{float(np.abs(a0 - a1).max()):.1e}"
            sm = pg["speed_ms"]
            print(f"{s:<14}{pg['page']:<26}{pg['shapes']:>9}{pg['first_s']:>10.1f}{pg['steady_s']:>10.1f}"
                  f"{sm['preprocess']:>9.0f}{sm['inference']:>10.0f}{sm['postprocess']:>9.0f}{sp:>9}{dp:>10}")
        print(f"{'':<14}load {res['load_s']:.1f} s, threads {res['threads']}, torch {res['torch']}")


if __name__ == "__main__":
    main()
