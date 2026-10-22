"""Ping a Colab VM every 4 minutes while its jobs run (two VMs were reclaimed after ~25 idle minutes, LEDGER section 4),
print the last progress line, and download a pretraining checkpoint as soon as that run has finished.

    py -3.12 tools/colab_keepalive.py VM [PRETRAIN_RUN] [--backup RUN ...]   # exits on jobs.done or when the VM is gone

``--backup RUN``: every finished ``runs/RUN/fold<k>`` (its ``result.json``, ``val_scores.npz``) and ``args.json`` are copied to
``runs_colab/RUN/`` as soon as they appear, so a reclaimed VM loses at most the running fold (rerun with ``resume=``).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cx  # noqa: E402

PROBE = """import os, glob, json
logs = sorted(glob.glob('/content/job*.log'))
last = []
if logs:
    t = open(logs[-1], errors='replace').read().replace(chr(13), chr(10)).splitlines()
    last = [l.strip()[:110] for l in t if 'it/s' in l][-1:]
pre = %r
print('@@CX@@' + json.dumps({'jobs': len(logs), 'last': last, 'done': os.path.exists('/content/jobs.done'),
                             'pre': bool(pre) and os.path.exists('/content/runs/%%s/fold0/result.json' %% pre)}))"""


def backup(vm, run, have):
    files = cx.exec_json(vm, f"""import glob, os, json
fs = sorted(glob.glob('/content/runs/{run}/fold*/result.json'))
out = [f.replace('result.json', n) for f in fs for n in ('result.json', 'val_scores.npz') if os.path.exists(f.replace('result.json', n))]
out += glob.glob('/content/runs/{run}/args.json')
print('@@CX@@' + json.dumps({{f: os.path.getsize(f) for f in out}}))""", timeout=240)
    for f, size in files.items():
        if f in have:
            continue
        cx.get(vm, f, Path("runs_colab") / f.split("/content/runs/", 1)[1], size=size)
        have.add(f)
        print("backed up", f, flush=True)


def main(vm, pre="", backups=()):
    got = not pre
    have, fails = set(), 0
    while True:
        try:
            r = cx.exec_json(vm, PROBE % pre, timeout=240)
            print(time.strftime("%H:%M"), json.dumps(r, ensure_ascii=False), flush=True)
            if r["pre"] and not got:
                remote = f"/content/runs/{pre}/fold0/best.pt"
                size = cx.exec_json(vm, f"import os;print({cx.MARK!r}+str(os.path.getsize({remote!r})))")
                cx.get(vm, remote, Path("runs_colab") / pre / "fold0" / "best.pt", size=size)
                got = True
                print("saved", pre, "checkpoint locally", flush=True)
            for run in backups:
                backup(vm, run, have)
            fails = 0
            if r["done"]:
                print("JOBS DONE", flush=True)
                return 0
        except Exception as e:
            fails += 1
            print(time.strftime("%H:%M"), "ERR", repr(e)[:300], flush=True)
            if fails >= 4 and not cx.alive(vm):  # the session listing alone fails now and then: never trust it
                print("VM GONE", flush=True)
                return 1
        time.sleep(240)


if __name__ == "__main__":
    argv = sys.argv[1:]
    bk = argv[argv.index("--backup") + 1:] if "--backup" in argv else []
    pos = argv[: argv.index("--backup")] if "--backup" in argv else argv
    sys.exit(main(pos[0], pos[1] if len(pos) > 1 else "", bk))
