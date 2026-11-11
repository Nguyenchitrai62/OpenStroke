"""Small asynchronous timing sample of existing train steps; never adds a batch or changes RNG."""
import time
import torch


class StepTimer:
    def __init__(self, device, samples=5, skip=5):
        self.cuda = torch.device(device).type == "cuda"
        self.samples, self.skip = samples, skip
        self.points = None
        self.traces = []

    def begin(self, step):
        self.points = [] if self.skip <= step < self.skip+self.samples else None
        self.mark("start")

    def mark(self, phase):
        if self.points is None:
            return
        if self.cuda:
            point = torch.cuda.Event(enable_timing=True)
            point.record()
        else:
            point = time.perf_counter()
        self.points.append((phase,point))

    def finish(self):
        if self.points is None:
            return
        self.mark("optimizer")
        self.traces.append(self.points)
        self.points = None

    def summary(self):
        sums = {}
        for points in self.traces:
            if self.cuda:
                points[-1][1].synchronize()  # queried after the epoch, not on the measured critical path
            for (_,a),(phase,b) in zip(points,points[1:]):
                value = a.elapsed_time(b) if self.cuda else 1000*(b-a)
                sums[phase] = sums.get(phase,0.)+value
        count=len(self.traces)
        return {k:round(v/max(count,1),3) for k,v in sums.items()},count
