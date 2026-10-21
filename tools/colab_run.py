"""Run ``openstroke train`` jobs on a Colab VM from this machine (no notebook): the agent's own Colab loop.

    py -3.12 tools/colab_run.py start VM ZIP CMD [CMD ...]   # new G4 VM (if absent), upload + unzip ZIP, install main, run CMDs in turn
    py -3.12 tools/colab_run.py poll VM                      # val F2 per epoch of every log, last val table, GPU, CU balance
    py -3.12 tools/colab_run.py fetch VM                     # logs + runs/ (best.pt, val_scores.npz, summaries) -> runs_colab/VM.tgz
    py -3.12 tools/colab_run.py stop VM                      # ALWAYS stop when done (G4 ~8.9 CU/h)

CMDs are run from /content; data unzips to /content/dataset_local (as the user's notebook). Logs: /content/job<i>.log.
Push before ``start``: the VM installs the code from GitHub ``main``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cx  # noqa: E402

M = cx.MARK


def start(vm, zip_path, *cmds):
    if vm not in cx.sessions():
        cx.new(vm, "G4")
    remote = f"/content/{Path(zip_path).name}"
    cx.put(vm, zip_path, remote)
    out = cx.exec_py(vm, f"""import subprocess
print(subprocess.run('mkdir -p /content/dataset_local && cd /content/dataset_local && unzip -q -o {remote} && '
                     'pip install -q --force-reinstall --no-deps git+https://github.com/Nguyenchitrai62/OpenStroke.git 2>&1 | tail -2; '
                     'pip install -q git+https://github.com/Nguyenchitrai62/OpenStroke.git 2>&1 | tail -2; '
                     'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader',
                     shell=True, capture_output=True, text=True).stdout)
print({M!r}+'ok')""", timeout=1500)[1]
    print(out[-800:])
    chain = "; ".join(f"{c} > /content/job{i}.log 2>&1" for i, c in enumerate(cmds))
    job = f"cd /content && ({chain}; touch /content/jobs.done) &"
    cx.exec_py(vm, f"""import subprocess
subprocess.Popen({job!r}, shell=True, start_new_session=True)
print({M!r}+'ok')""", timeout=300)
    print(f"started {len(cmds)} job(s) on {vm}")


def launch(vm, *cmds):
    """More jobs on a VM that already has the data (after its previous jobs are done): job<i> numbering continues."""
    n = cx.exec_json(vm, f"import glob;print({M!r}+str(len(glob.glob('/content/job*.log'))))")
    chain = "; ".join(f"{c} > /content/job{n + i}.log 2>&1" for i, c in enumerate(cmds))
    job = f"cd /content && rm -f /content/jobs.done && ({chain}; touch /content/jobs.done) &"
    cx.exec_py(vm, f"""import subprocess
subprocess.Popen({job!r}, shell=True, start_new_session=True)
print({M!r}+'ok')""", timeout=300)
    print(f"launched {len(cmds)} job(s) on {vm} as job{n}..")


def poll(vm):
    print(cx.exec_py(vm, """import subprocess, os, glob
for f in sorted(glob.glob('/content/job*.log')):
    t = open(f, errors='replace').read().replace('\\r', '\\n').splitlines()
    val = [l for l in t if l.strip().startswith('all ') and len(l.split()) > 6]
    print('==', f, 'val F2 per epoch:', ' '.join(l.split()[6] for l in val))
    print('\\n'.join([l for l in t if 'it/s' in l][-1:] + [l for l in t if 'it/s' not in l][-14:]))
print('done' if os.path.exists('/content/jobs.done') else 'running')
print(subprocess.run('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader', shell=True,
                     capture_output=True, text=True).stdout)""", timeout=300)[1][-7000:])
    print("CU balance / rate:", cx.balance())


def fetch(vm):
    cx.exec_py(vm, "import subprocess;subprocess.run('cd /content && tar czf out.tgz job*.log runs "
                   "--exclude=last.pt --exclude=cache', shell=True)", timeout=900)
    size = cx.exec_json(vm, f"import os;print({M!r}+str(os.path.getsize('/content/out.tgz')))")
    dst = Path("runs_colab") / f"{vm}.tgz"
    cx.get(vm, "/content/out.tgz", dst, size=size)
    print("saved", dst, size)


def finish(vm, run, job="0"):
    """Final TTA table of job<job>, its val_scores.npz / result.json -> runs_colab/<run>/fold0/, then stop the VM."""
    code = ("t=open('/content/job%s.log',errors='replace').read().replace(chr(13),chr(10)).splitlines()\n"
            "i=[k for k,l in enumerate(t) if l.startswith('Fold 1 best')]\n"
            "print(chr(10).join(t[i[0]:i[0]+18]) if i else 'no final table')" % job)
    out = cx.exec_py(vm, code, timeout=300)[1]
    print(out[-2500:])
    for f in ("val_scores.npz", "result.json"):
        r = f"/content/runs/{run}/fold0/{f}"
        size = cx.exec_json(vm, f"import os;print({M!r}+str(os.path.getsize({r!r})))")
        cx.get(vm, r, Path("runs_colab") / run / "fold0" / f, size=size)
    cx.stop(vm)
    print("stopped", vm, "CU balance / rate:", cx.balance())


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"start": start, "launch": launch, "poll": poll, "finish": finish, "fetch": fetch, "stop": cx.stop}[cmd](*rest)
