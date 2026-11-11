"""Portable product calibration; no dataset paths or external sidecars needed for new checkpoints."""
import math


def object_calibration(checkpoint):
    score = checkpoint.get("object_score") or {}
    args = checkpoint.get("args", {})
    mode = "instances" if args.get("instance_queries") else "graph" if args.get("stroke_graph") else "members"
    if mode in score:
        score = score[mode] or {}
    if not (args.get("stroke_graph") or args.get("arch") == "v25") or "best_conf" not in score:
        return None
    clean = {k: float(score[k]) for k in ("best_conf", "F2", "P", "R") if k in score}
    if not all(math.isfinite(v) for v in clean.values()) or not 0 <= clean["best_conf"] <= 1:
        raise ValueError("Invalid checkpoint object calibration")
    if isinstance(score.get("protocol"), str):
        clean["protocol"] = score["protocol"]
    return clean
