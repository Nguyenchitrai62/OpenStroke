"""Compact YOLO-style training console (layout of OpenOBB ``console.py``): one in-place progress row per
epoch, then one validation row. Everything else goes to ``train.log``."""

from __future__ import annotations

import sys
import time

W = 12


def _bar(i, n, width=12):
    k = int(width * i / max(n, 1))
    return "━" * k + "─" * (width - k)


def _rate(i, dt):
    if i <= 0 or dt <= 0:
        return ""
    r = i / dt
    return f"{r:.1f}it/s" if r >= 1 else f"{1 / r:.1f}s/it"


def dur(s):
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s // 60)}:{int(s % 60):02d}"
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


class EpochBar:
    """bar = EpochBar(epoch, epochs, n_iters, cols); bar.update(i, means, shapes, mem); bar.close()."""

    def __init__(self, epoch, epochs, n, cols, window, stream=None, every=1.0, header=True):
        self.epoch, self.epochs, self.n, self.cols, self.window = epoch, epochs, n, cols, window
        stream = stream or sys.stdout  # looked up per call: a default bound at import outlives a replaced stdout
        self.stream, self.every = stream, every
        self.t0 = self.last = time.time()
        self.row = ""
        if header:
            print("\n" + f"{'Epoch':>{W}}{'GPU_mem':>{W}}" + "".join(f"{c + '_loss':>{W}}" for c in cols)
                  + f"{'Shapes':>{W}}{'Window':>{W}}", file=stream, flush=True)

    def update(self, i, means, shapes, mem_gb, force=False):
        now = time.time()
        if not force and now - self.last < self.every and i < self.n:
            return
        self.last = now
        vals = "".join(f"{means.get(k, 0.0):>{W}.4f}" for k in self.cols)
        mem = f"{mem_gb:.1f}G" if mem_gb is not None else "-"
        self.row = (f"{f'{self.epoch + 1}/{self.epochs}':>{W}}{mem:>{W}}{vals}{shapes:>{W},}{self.window:>{W}}: "
                    f"{100 * i / max(self.n, 1):3.0f}% {_bar(i, self.n)} {i}/{self.n} {_rate(i, now - self.t0)} "
                    f"{dur(now - self.t0)}")
        self.stream.write("\r" + self.row)
        self.stream.flush()

    def close(self):
        self.stream.write("\n")
        self.stream.flush()


VAL_HEAD = (f"{'':>{W}}{'Pages':>{W}}{'Shapes':>{W}}{'True(pt)':>{W}}{'P':>{W}}{'R':>{W}}{'F2':>{W}}"
            f"{'F2@0.5':>{W}}{'thr':>{W}}")


def val_row(name, pages, shapes, true_pt, p, r, f2, f2_05, thr):
    return (f"{name:>{W}}{pages:>{W}}{shapes:>{W},}{true_pt:>{W},.0f}{p:>{W}.3f}{r:>{W}.3f}{f2:>{W}.3f}"
            f"{f2_05:>{W}.3f}{thr:>{W}.2f}")


def table(rows: list[list], heads: list[str], widths: list[int] | None = None) -> str:
    widths = widths or [max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(heads)]
    line = "".join(f"{h:>{w}}" for h, w in zip(heads, widths))
    return "\n".join([line] + ["".join(f"{str(c):>{w}}" for c, w in zip(r, widths)) for r in rows])
