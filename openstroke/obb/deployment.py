"""Portable product calibration; no dataset paths or external sidecars needed for new checkpoints."""
import math


def object_calibration(checkpoint):
    score = checkpoint.get("object_score") or {}
    if "graph" in score:
        score = score["graph"] or {}
    if not checkpoint.get("args", {}).get("stroke_graph") or "best_conf" not in score:
        return None
    clean = {k: float(score[k]) for k in ("best_conf", "F2", "P", "R") if k in score}
    if not all(math.isfinite(v) for v in clean.values()) or not 0 <= clean["best_conf"] <= 1:
        raise ValueError("Invalid checkpoint object calibration")
    return clean
