"""Command line: ``py -3.12 -m openstroke <command>``.

Commands:
  build-e0   GT + payload -> datasets/e0/*.npz + manifest.json (reads AI_Takeoff read-only)
  qa-e0      manifest.json -> research/E0_QA.md
  folds-propose  draft datasets/e0/folds_proposal.json (the user confirms it; AGENTS.md 3.4)
  baseline-layer E1.1 layer-selection ceiling (numpy, local)
  baseline-gbdt  E1.3 gradient boosting on hand features (needs scikit-learn; Colab)
  train      model v1 cross-validation (needs torch)
  bench      time one training step + peak GPU memory for batch / window choices
  pack       zip E0 data (npz + manifest + folds) for Colab
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .sources import DEFAULT_AI_TAKEOFF_BE


_KV = re.compile(r"^([A-Za-z_][\w-]*)=(.*)$")
_FALSE = ("false", "0", "no", "off")
_TRUE = ("true", "1", "yes", "on")
#: key=value names whose boolean form is a ``--no-...`` / ``--...`` switch
_SWITCHES = {"layer-pool": ("--no-layer-pool", None), "exist-ok": (None, "--exist-ok")}


def yolo_args(argv: list[str]) -> list[str]:
    """YOLO / OpenOBB style ``key=value`` -> ``--key value``, so both spellings work:

    ``openstroke train data=e0 epochs=30 layer_pool=false`` == ``openstroke train --data e0 --epochs 30 --no-layer-pool``
    """

    out: list[str] = []
    for tok in argv:
        m = _KV.match(tok)
        if not m or tok.startswith("-"):
            out.append(tok)
            continue
        key, val = m.group(1).replace("_", "-"), m.group(2)
        low = val.lower()
        if key in _SWITCHES:
            off, on = _SWITCHES[key]
            flag = off if low in _FALSE else on if low in _TRUE else None
            if flag is None and low not in _FALSE + _TRUE:
                raise SystemExit(f"{key}={val}: expected true or false")
            if flag:
                out.append(flag)
            continue
        out += [f"--{key}", val]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="openstroke")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-e0", help="build the E0 dataset")
    b.add_argument("--ai-takeoff-be", type=Path, default=DEFAULT_AI_TAKEOFF_BE)
    b.add_argument("--out", type=Path, default=Path("datasets/e0"))
    b.add_argument("--only", nargs="*", help="stems or document ids to build (debug); the manifest then lists only these")
    q = sub.add_parser("qa-e0", help="write the E0 QA report")
    q.add_argument("--manifest", type=Path, default=Path("datasets/e0/manifest.json"))
    q.add_argument("--out", type=Path, default=Path("research/E0_QA.md"))
    f = sub.add_parser("folds-propose", help="draft a fold file grouped by PDF")
    f.add_argument("--data", type=Path, default=Path("datasets/e0"))
    f.add_argument("--k", type=int, default=5)
    f.add_argument("--out", type=Path, default=Path("datasets/e0/folds_proposal.json"))
    bl = sub.add_parser("baseline-layer", help="E1.1 layer-selection ceiling")
    bl.add_argument("--data", type=Path, default=Path("datasets/e0"))
    bl.add_argument("--folds", type=Path, default=Path("datasets/e0/folds_proposal.json"))
    bl.add_argument("--out", type=Path, default=Path("runs/e1_layer_ceiling.json"))
    bg = sub.add_parser("baseline-gbdt", help="E1.3 gradient boosting baseline")
    bg.add_argument("--data", type=Path, default=Path("datasets/e0"))
    bg.add_argument("--folds", type=Path, default=Path("datasets/e0/folds.json"))
    bg.add_argument("--out", type=Path, default=Path("runs/e1_gbdt"))
    bg.add_argument("--cache", type=Path, default=None)
    bg.add_argument("--max-train", type=int, default=3_000_000)
    tr = sub.add_parser("train", help="train model v1 (cross-validation by PDF)")
    from .train import add_args

    add_args(tr)
    bn = sub.add_parser("bench", help="time a training step and peak GPU memory (synthetic data)")
    bn.add_argument("--model", default="openstroke2")
    bn.add_argument("--batch", default="4", help="comma-separated, e.g. 4,8,16")
    bn.add_argument("--window", default="65536", help="comma-separated")
    bn.add_argument("--iters", type=int, default=5)
    bn.add_argument("--device", default="auto")
    bn.add_argument("--dims", default="64,96,128,192,256")
    bn.add_argument("--enc-depths", default="2,2,2,4,2")
    bn.add_argument("--dec-depths", default="1,1,1,1")
    bn.add_argument("--patch", type=int, default=1024)
    pk = sub.add_parser("pack", help="zip E0 data for Colab")
    pk.add_argument("--data", type=Path, default=Path("datasets/e0"))
    pk.add_argument("--folds", type=Path, default=Path("datasets/e0/folds_proposal.json"))
    pk.add_argument("--out", type=Path, default=Path("datasets/openstroke_e0_colab.zip"))
    argv = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(argv[:1] + yolo_args(argv[1:]))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.cmd == "build-e0":
        from .build import build_all

        m = build_all(args.ai_takeoff_be, args.out, only=set(args.only) if args.only else None,
                      log=lambda s: print(s, flush=True))
        ok = sum(r["status"] == "ok" for r in m["records"])
        print(f"wrote {args.out / 'manifest.json'}: {ok}/{len(m['records'])} GT pages ok, "
              f"{len(m['physical_pages'])} physical pages")
        return 0 if ok == len(m["records"]) else 1
    if args.cmd == "qa-e0":
        from .qa import write_report

        write_report(args.manifest, args.out)
        print(f"wrote {args.out}")
        return 0
    if args.cmd == "folds-propose":
        import json

        from .folds import propose

        manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
        args.out.write_text(json.dumps(propose(manifest, args.k), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.out} (status: proposal)")
        return 0
    if args.cmd == "baseline-layer":
        import json

        from .baseline import layer_ceiling
        from .folds import load, resolve

        manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
        entries, warn = resolve(manifest, load(args.folds))
        for w in warn:
            print("WARNING:", w)
        res = layer_ceiling(args.data, [s for s, _, _ in entries])
        modes = {s: m for s, _, m in entries}
        for p in res["pages"]:
            print(f"{p['stem']} {modes[p['stem']]}: layers {p['chosen_layers']}/{p['layers']} R {p['recall']:.3f} "
                  f"P {p['precision']:.3f} F2 {p['f2']:.3f}")
        print(f"ALL: R {res['all']['recall']:.3f} P {res['all']['precision']:.3f} F2 {res['all']['f2']:.3f}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0
    if args.cmd == "baseline-gbdt":
        from .baseline import gbdt

        gbdt(args.data, args.folds, args.out, cache=args.cache, max_train=args.max_train)
        return 0
    if args.cmd == "train":
        from .train import main_train

        return main_train(args)
    if args.cmd == "bench":
        from .bench import bench

        ints = lambda v: [int(x) for x in str(v).split(",")]  # noqa: E731
        cfg = {} if args.model == "openstroke1" else dict(dims=ints(args.dims), enc_depths=ints(args.enc_depths),
                                                          dec_depths=ints(args.dec_depths), patch=args.patch)
        bench(args.model, ints(args.batch), ints(args.window), iters=args.iters, device=args.device, **cfg)
        return 0
    if args.cmd == "pack":
        from .pack import pack

        out = pack(args.data, args.folds, args.out)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
