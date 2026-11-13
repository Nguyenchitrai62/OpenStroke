"""A page's windows in several processes on CPU. The windows are independent and their sum does not depend on the order, so the answer stays the same.

Processes, not threads: windows in threads over one model died with an access violation inside torch at 4 threads (image branch, LEDGER F53). Each process loads the model
once and stays. A process that dies or fails, usually for lack of memory (one window can ask for 1.6 GB at once), only makes the page fall back to the sequential path (``predict`` returns None); it does not take the app down.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from pathlib import Path

import numpy as np

READY = "ready"


def _serve(checkpoint: str, threads: int, tasks, results) -> None:
    import torch

    from .data import page_from_npz
    from .engine.trainer import score_items
    from .infer import Predictor

    torch.set_flush_denormal(True)
    pred = Predictor(checkpoint, device="cpu", threads=threads)
    results.put((READY, None, None))
    parent = mp.parent_process()
    current, page = None, None
    while parent is None or parent.is_alive():  # a killed parent must not leave a process holding the model in memory
        try:
            job, npz, mode, graph, item = tasks.get(timeout=2)
        except queue.Empty:
            current, page = None, None  # idle: give the page's memory back
            continue
        try:
            if job != current:
                page, current = page_from_npz(Path(npz), "page", mode, graph=graph,
                                              no_layers=bool(pred.config.get("no_layers"))), job
            pos, prob, _ = score_items(pred.model, page, [item], pred.window, pred.device)
            results.put((job, pos.astype(np.int32), prob.astype(np.float32)))
        except Exception as e:  # noqa: BLE001  # reported, the page falls back
            results.put((job, None, repr(e)))


class WindowPool:
    def __init__(self, checkpoint: str | Path, workers: int, threads: int):
        ctx = mp.get_context("spawn")
        self.tasks, self.results = ctx.Queue(), ctx.Queue()
        per = max(1, threads // workers)
        self.procs = [ctx.Process(target=_serve, args=(str(checkpoint), per, self.tasks, self.results), daemon=True) for _ in range(workers)]
        for p in self.procs:
            p.start()
        self.ready = 0
        self.job = 0
        self.error: str | None = None

    def _get(self):
        """Next result; None when a process has died (no result can be trusted to come)."""

        while True:
            try:
                r = self.results.get(timeout=1)
            except queue.Empty:
                if not all(p.is_alive() for p in self.procs):
                    self.error = "a window process died"
                    return None
                continue
            if r[0] == READY:
                self.ready += 1
                continue
            return r

    def wait_ready(self) -> bool:
        while self.ready < len(self.procs):
            try:
                if self.results.get(timeout=1)[0] == READY:
                    self.ready += 1
            except queue.Empty:
                if not all(p.is_alive() for p in self.procs):
                    self.error = "a window process died while loading"
                    return False
        return True

    def predict(self, page, npz: str | Path, mode: str, graph: bool, window_size: int, tta: int = 1, on_batch=None) -> np.ndarray | None:
        """Probabilities as ``predict_page``, or None when this pool cannot be used any more."""

        from .engine.trainer import eval_windows

        if self.error or not self.wait_ready():
            return None
        self.job += 1
        items = [(s, k) for s in eval_windows(page, window_size) for k in range(tta)]
        for item in items:
            self.tasks.put((self.job, str(npz), mode, graph, item))
        total, count = np.zeros(len(page), np.float64), np.zeros(len(page), np.float64)
        done = 0
        while done < len(items):
            r = self._get()
            if r is None:
                return None
            job, pos, prob = r
            if job != self.job:  # left over from a page that fell back
                continue
            if pos is None:
                self.error = prob
                return None
            np.add.at(total, pos, prob)
            np.add.at(count, pos, 1.0)
            done += 1
            if on_batch is not None:
                on_batch(done, len(items))
        return total / np.maximum(count, 1)

    def close(self) -> None:
        for p in self.procs:
            p.terminate()
        deadline = time.time() + 5
        for p in self.procs:
            p.join(max(0.0, deadline - time.time()))
