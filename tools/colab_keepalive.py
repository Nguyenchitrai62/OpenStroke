"""Ping a Colab VM every 4 minutes while its jobs run (two VMs were reclaimed after ~25 idle minutes, LEDGER section 4),
print the last progress line, and download a pretraining checkpoint as soon as that run has finished.

    py -3.12 tools/colab_keepalive.py VM [PRETRAIN_RUN]     # exits on jobs.done or when the VM is gone
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


def main(vm, pre=""):
    got = not pre
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
            if r["done"]:
                print("JOBS DONE", flush=True)
                return 0
        except Exception as e:
            print(time.strftime("%H:%M"), "ERR", repr(e)[:300], flush=True)
            if vm not in cx.sessions():
                print("VM GONE", flush=True)
                return 1
        time.sleep(240)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
