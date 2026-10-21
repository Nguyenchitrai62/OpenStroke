"""E8 on Colab without the notebook: G4 VM, slim e3 data (FIRE / HVAC only), E8a then E8b in the background.

    py -3.12 tools/colab_e8.py start     # new VM + upload + install + launch (returns once the runs are started)
    py -3.12 tools/colab_e8.py poll      # tail of both logs, GPU, CU balance
    py -3.12 tools/colab_e8.py fetch     # download logs + summaries + best.pt into runs_colab/
    py -3.12 tools/colab_e8.py stop      # stop the VM (always, when done)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cx  # noqa: E402

VM = "e8-g4"
ZIP = Path("datasets/openstroke_e3fh_colab.zip")
D = "/content/dataset_local/e3"
COMMON = "fold=0 modes=fire,hvac feature_drop=layer_stats batch=4"
E8A = f"openstroke train model=openstroke2 data={D} folds={D}/folds_dev_pseudo.json {COMMON} epochs=100 lr0=0.002 name=e8a"
E8B = f"openstroke train model=runs/e8a data={D} folds={D}/folds_dev.json {COMMON} epochs=30 lr0=0.0005 name=e8b"
M = cx.MARK


def start():
    if VM not in cx.sessions():
        cx.new(VM, "G4")
    cx.put(VM, ZIP, "/content/e3fh.zip")
    out = cx.exec_py(VM, f"""import subprocess
print(subprocess.run('mkdir -p /content/dataset_local && cd /content/dataset_local && unzip -q -o /content/e3fh.zip && ls e3 | wc -l && '
                     'pip install -q git+https://github.com/Nguyenchitrai62/OpenStroke.git 2>&1 | tail -2 && '
                     'openstroke --help | head -3 && nvidia-smi --query-gpu=name,memory.total --format=csv',
                     shell=True, capture_output=True, text=True))
print({M!r}+'ok')""", timeout=1500)[1]
    print(out[-1500:])
    job = f"cd /content && ({E8A} > /content/e8a.log 2>&1; {E8B} > /content/e8b.log 2>&1; touch /content/e8.done) &"
    print(cx.exec_py(VM, f"""import subprocess
subprocess.Popen({job!r}, shell=True, start_new_session=True)
print({M!r}+'ok')""", timeout=300)[1][-300:])


def poll():
    print(cx.exec_py(VM, """import subprocess, os
for f in ('/content/e8a.log', '/content/e8b.log'):
    if os.path.exists(f):
        t = open(f, errors='replace').read().replace('\\r', '\\n').splitlines()
        keep = [l for l in t if l.strip().startswith(('all ', 'Stopping', 'Fold', 'SUMMARY', 'Traceback', 'Error'))
                or ' all ' in l[:20] or '/100 ' in l or '/30 ' in l or 'Error' in l]
        print('==', f, len(t), 'lines'); print('\\n'.join(keep[-12:])); print('\\n'.join(t[-3:]))
print('done' if os.path.exists('/content/e8.done') else 'running')
print(subprocess.run('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader', shell=True,
                     capture_output=True, text=True).stdout)""", timeout=300)[1][-6000:])
    print("CU balance / rate:", cx.balance())


def fetch():
    out = Path("runs_colab")
    cx.exec_py(VM, "import subprocess;subprocess.run('cd /content && tar czf e8.tgz e8a.log e8b.log runs/e8a runs/e8b "
                   "--exclude=last.pt --exclude=val_scores.npz --exclude=cache', shell=True)", timeout=600)
    size = cx.exec_json(VM, f"import os;print({M!r}+str(os.path.getsize('/content/e8.tgz')))")
    cx.get(VM, "/content/e8.tgz", out / "e8.tgz", size=size)
    print("saved", out / "e8.tgz", size)


if __name__ == "__main__":
    {"start": start, "poll": poll, "fetch": fetch, "stop": lambda: cx.stop(VM)}[sys.argv[1]]()
