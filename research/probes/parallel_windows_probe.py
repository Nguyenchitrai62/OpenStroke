"""How much faster are a page's windows in several processes on CPU (``openstroke.parallel``), and is the answer the same? One real page.

    py -3.12 research/probes/parallel_windows_probe.py runs_colab/e37/fold0/best.pt datasets/e3/doc000194_p0001_gt121.npz 8
"""

import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(model, npz, cores):
    import torch

    from openstroke.data import page_from_npz
    from openstroke.engine.trainer import predict_page
    from openstroke.infer import Predictor

    cores = int(cores)
    torch.set_flush_denormal(True)
    pred = Predictor(model, device="cpu", threads=cores)
    page = page_from_npz(Path(npz), "page", "fire", graph=False)
    t0 = time.time()
    base = predict_page(pred.model, page, pred.window, pred.device, batch=1)
    print(f"1 process, {cores} threads: {len(page)} shapes, {time.time() - t0:.1f} s", flush=True)
    for workers in (2, 4):
        pool = Predictor(model, device="cpu", threads=cores, workers=workers).pool
        pool.wait_ready()
        t0 = time.time()
        p = pool.predict(page, npz, "fire", False, pred.window)
        rss = sum(psutil.Process(q.pid).memory_info().rss for q in pool.procs) / 2**30
        print(f"{workers} processes, {cores // workers} threads each: {time.time() - t0:.1f} s, largest difference {np.abs(p - base).max():.2e}, their memory {rss:.1f} GB", flush=True)
        pool.close()


if __name__ == "__main__":
    main(*sys.argv[1:4])
