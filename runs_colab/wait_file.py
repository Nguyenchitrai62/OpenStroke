"""Wait for jobs.done on a VM, fetch files, stop the VM.

    py -3.12 runs_colab/wait_file.py VM HOURS REMOTE=LOCAL [REMOTE=LOCAL ...]

Git Bash rewrites "/content/..." arguments to "C:/Program Files/Git/content/..." (lost E21, and the first E66 waiters): undone here.
"""
import re, sys, time
sys.path.insert(0, 'tools'); import cx
from pathlib import Path
vm, hours = sys.argv[1], float(sys.argv[2])
pairs = [re.sub(r"^[A-Za-z]:/Program Files/Git(?=/)", "", a.replace("\\", "/")).split("=", 1) for a in sys.argv[3:]]
t0 = time.time()
while time.time() - t0 < hours * 3600:
    try:
        rc, out = cx.exec_py(vm, "import os,glob\nprint('@@CX@@'+str(os.path.exists('/content/jobs.done'))+str(bool(glob.glob('/content/job*.log'))))", timeout=120)
    except Exception as e:
        print(time.strftime('%H:%M'), "ERR", str(e)[:100], flush=True); time.sleep(60); continue
    if "@@CX@@FalseFalse" in out and time.time() - t0 > 20 * 60:  # no job ever started (E76: a failed launch left the VM idle ~2 h)
        cx.stop(vm); print(vm, "no job log after 20 min: stopped", cx.balance(), flush=True)
        break
    if "@@CX@@True" in out:  # jobs.done
        for remote, local in pairs:
            try:
                size = cx.exec_json(vm, f"import os;print('@@CX@@'+str(os.path.getsize({remote!r})))")
                Path(local).parent.mkdir(parents=True, exist_ok=True)
                cx.get(vm, remote, local, size=size)
                print("fetched", remote, "->", local, Path(local).stat().st_size, flush=True)
            except Exception as e:
                print("could not fetch", remote, str(e)[:200], flush=True)
        cx.stop(vm); print(vm, "stopped", cx.balance(), flush=True)
        break
    print(time.strftime('%H:%M'), "waiting", flush=True)
    time.sleep(120)
else:
    cx.stop(vm)  # a run past its time budget is stopped, never left running
    print("timeout: VM stopped", vm, cx.balance(), flush=True)
