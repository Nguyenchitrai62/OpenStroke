"""Stop a Colab run early when its val curve is clearly worse than the best run's at the same epoch (user 2026-11-07: read the log,
cancel the session as soon as a run cannot reach the goal or beat the best version; save CU).

    py -3.12 runs_colab/early_stop.py VM RUN REF_LOG [--job 1] [--margin 0.02] [--patience 3] [--min-epoch 5]

Every 3 minutes reads /content/job<job>.log on the VM, takes mAP50 of each val line (``all  <images> <instances> P R mAP50 mAP50-95``) and
compares it with the same epoch of REF_LOG (the best run's train.log, same epoch schedule). Stops the VM (after fetching the log to
runs_colab/RUN/train_stopped.log) when, from --min-epoch on, the run is below the reference by more than --margin for --patience epochs
in a row, or when at 60 % of the epochs its best so far is below the reference's best by more than 2 x margin. Exits when jobs.done.
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "tools")
import cx  # noqa: E402

VAL = re.compile(r"^\s+all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
EPOCH = re.compile(r"^\s+(\d+)/(\d+)\s")


def curve(text: str) -> tuple[list[float], int]:
    vals, total = [], 0
    for line in text.replace("\r", "\n").splitlines():
        m = EPOCH.match(line)
        if m:
            total = int(m.group(2))
        m = VAL.match(line)
        if m:
            vals.append(float(m.group(3)))
    return vals, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vm")
    ap.add_argument("run")
    ap.add_argument("ref")
    ap.add_argument("--job", default="1")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-epoch", type=int, default=5)
    a = ap.parse_args()
    ref, _ = curve(Path(a.ref).read_text(encoding="utf-8", errors="replace"))
    log = f"/content/job{a.job}.log"
    while True:
        try:
            done = "@@CX@@True" in cx.exec_py(a.vm, "import os\nprint('@@CX@@'+str(os.path.exists('/content/jobs.done')))", timeout=120)[1]
            text = cx.exec_py(a.vm, f"import os\nprint(open({log!r},errors='replace').read() if os.path.exists({log!r}) else '')", timeout=300)[1]
        except Exception as e:
            print(time.strftime("%H:%M"), "ERR", str(e)[:120], flush=True)
            if a.vm not in cx.sessions():
                print("VM gone; exit", flush=True)
                return 0
            time.sleep(60)
            continue
        vals, total = curve(text)
        n = len(vals)
        below = 0
        for e in range(n):
            below = below + 1 if e + 1 >= a.min_epoch and e < len(ref) and vals[e] < ref[e] - a.margin else 0
        late = total and n >= 0.6 * total and max(vals) < max(ref[:total] or ref) - 2 * a.margin
        print(time.strftime("%H:%M"), f"epoch {n}/{total}", "run", [round(v, 3) for v in vals[-3:]], "ref", [round(v, 3) for v in ref[max(n - 3, 0):n]],
              f"below {below}", flush=True)
        finished = total and n >= total  # training is over (final eval): the result fetcher takes it from here
        if (below >= a.patience or late) and not finished:
            why = f"{below} epochs in a row below the reference by > {a.margin}" if below >= a.patience else "best so far cannot reach the reference"
            out = Path("runs_colab") / a.run / "train_stopped.log"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            cx.stop(a.vm)
            print("STOPPED", a.vm, "-", why, "; balance", cx.balance(), flush=True)
            return 0
        if done or finished:
            print("jobs done", flush=True)
            return 0
        time.sleep(180)


if __name__ == "__main__":
    sys.exit(main())
