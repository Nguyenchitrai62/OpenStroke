"""OpenOBB25: vector tokens in, primitive membership of every box out (docs/OPENOBB25.md)."""
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from openstroke.obb.data.dota import DotaPatches, collate
from openstroke.obb.engine import ctx_batch
from openstroke.obb.models.openobb24 import OpenOBB24
from openstroke.obb.models.openobb25 import (INST_UNKNOWN, OpenOBB25, OpenOBB25Loss, balanced_pick, member_lists,
                                      pair_geometry)
from openstroke.obb.models.openobb3 import fast_nms

torch.set_num_threads(1)

IMG, NC = 128, 3
G = [0, 1, 1]


def _pair():
    torch.manual_seed(0)
    a = OpenOBB24("s", num_classes=NC, img_size=IMG, max_det=50, groups=G).eval()
    b = OpenOBB25("s", num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1).eval()
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith(("vec.", "member.")) for k in missing)
    return a, b


def _edge_tokens(box, inst, n_extra=0, seed=0):
    """Line primitives on the 4 sides of box (cx, cy, w, h, t) px -> rows (4, 24) model input (normalised by IMG)."""
    cx, cy, w, h, t = box
    corners = cv2.boxPoints(((cx, cy), (w, h), math.degrees(t)))
    rows = []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        pts = np.linspace(a, b, 8) / IMG
        rows.append(np.r_[0, pts.reshape(-1), 0, 0, 0, 1.0 / IMG, 0, inst, 100 * (inst + 1) + k])
    rng = np.random.default_rng(seed)
    for k in range(n_extra):                     # strays far from every object
        p0 = rng.uniform(0, 30, 2)
        pts = np.linspace(p0, p0 + 5, 8) / IMG
        rows.append(np.r_[1, pts.reshape(-1), 1, 0, 0, 1.0 / IMG, 0, -1, 900 + k])
    return np.array(rows, np.float32)


def _batch():
    boxes = [(60.0, 64.0, 70.0, 20.0, 0.0), (90.0, 100.0, 30.0, 30.0, 0.3)]
    vec = np.concatenate([_edge_tokens(boxes[0], 0), _edge_tokens(boxes[1], 1, n_extra=3)])
    tgt = {"labels": torch.tensor([0, 2]),
           "boxes": torch.tensor([[b[0] / IMG, b[1] / IMG, b[2] / IMG, b[3] / IMG, b[4]] for b in boxes]),
           "vec": torch.from_numpy(vec)}
    return [tgt, {k: v.clone() for k, v in tgt.items()}]


def test_v25_starts_as_v24():
    """Zero-initialised fusion: with or without vector tokens, the boxes and scores are OpenOBB24's at step 0."""
    a, b = _pair()
    x = torch.rand(2, 3, IMG, IMG)
    ctx = ctx_batch(_batch(), torch.device("cpu"))
    with torch.no_grad():
        oa = a(x)
        for c in (None, ctx):
            ob = b(x, ctx=c)
            assert torch.allclose(oa["pred_boxes"], ob["pred_boxes"], atol=1e-5)
            assert torch.allclose(oa["pred_logits"].sigmoid(), ob["pred_logits"].sigmoid(), atol=1e-5)
        assert "member_logits" in ob and ob["member_logits"].shape[:2] == ob["pred_logits"].shape[:2]
        assert "member_logits" not in b(x)


def test_member_loss_finite_and_decreases():
    torch.manual_seed(0)
    m = OpenOBB25("s", num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1).train()
    tg = _batch()
    x = torch.rand(2, 3, IMG, IMG)
    ctx = ctx_batch(tg, torch.device("cpu"))
    crit = OpenOBB25Loss(img_size=IMG, member_weight=1.0)
    opt = torch.optim.Adam(list(m.member.parameters()) + list(m.vec.parameters()), lr=3e-3)
    picks = m.member_targets(m(x, tg, ctx=ctx), tg, IMG)            # the real matching works on this batch...
    assert all(p is not None and len(p[0]) for p in picks)

    def gt_picks(out, targets, size):                               # ...then fixed boxes: no jitter noise in the test
        return [(t["boxes"] * torch.tensor([size] * 4 + [1.0]), torch.arange(len(t["boxes"]))) for t in targets]
    m.member_targets = gt_picks
    hist = []
    for step in range(12):
        out = m(x, [{k: v for k, v in t.items() if k != "vec"} for t in tg], ctx=ctx)
        losses = crit(out, tg)
        assert torch.isfinite(losses["member_bce"]) and torch.isfinite(losses["member_dice"])
        loss = losses["member_bce"] + losses["member_dice"]
        assert loss.item() > 0
        opt.zero_grad()
        loss.backward()
        if step == 0:
            assert m.member.t.weight.grad is not None and m.member.t.weight.grad.abs().sum() > 0
            assert m.vec.geo[0].weight.grad is not None                     # gradient reaches the token encoder
            assert all(t.grad is None for t in m.edge.parameters())         # box features are detached
        opt.step()
        hist.append(loss.item())
    assert hist[-1] < hist[0] * 0.8, hist


def test_balanced_pick_spreads_over_objects():
    best = torch.tensor([0.9, 0.95, 0.6, 0.55, 0.4, 0.7])
    gt = torch.tensor([0, 0, 1, 1, 1, 0])
    sel = balanced_pick(best, gt, 2)
    assert sorted(gt[sel].tolist()) == [0, 1]                  # one per object before the second of any
    assert set(sel.tolist()) == {1, 2}


def test_pair_geometry_inside_and_gate():
    box = torch.tensor([[50.0, 50.0, 40.0, 10.0, 0.0]])
    inside = torch.tensor([[[40.0 + i, 50.0] for i in range(8)]])
    far = torch.tensor([[[5.0 + i, 5.0] for i in range(8)]])
    g, gate = pair_geometry(box, torch.cat([inside, far]))
    assert g[0, 0, 0] == 1.0 and g[0, 1, 0] == 0.0
    assert bool(gate[0, 0]) and not bool(gate[0, 1])


def test_nms_keep_index_maps_rows_to_candidates():
    torch.manual_seed(0)
    logits = torch.full((1, 6, 2), -30.0)
    logits[0, :, 0] = torch.tensor([2.0, 1.9, -1.0, 0.5, 3.0, -30.0])
    boxes = torch.tensor([[[20, 20, 30, 6, 0.0], [20.5, 20, 30, 6, 0.0], [80, 80, 20, 8, 0.0],
                           [50, 60, 10, 10, 0.0], [100, 30, 40, 6, 0.5], [5, 5, 4, 4, 0.0]]])
    out_l, out_b, idx = fast_nms(logits, boxes, max_det=5, iou=0.5, return_index=True)
    n = int((idx[0] >= 0).sum())
    assert n == 4                                   # box 1 suppressed by box 0, box 5 below conf
    assert torch.allclose(out_b[0, :n], boxes[0, idx[0, :n]])
    assert idx[0, :n].tolist() == [4, 0, 3, 2]       # by score
    # through the model: row r of member_logits belongs to output row r (keep_index -> candidate)
    _, b = _pair()
    with torch.no_grad():
        o = b(torch.rand(2, 3, IMG, IMG), ctx=ctx_batch(_batch(), torch.device("cpu")))
    assert o["keep_index"].shape == o["pred_logits"].shape[:2]
    dead = o["keep_index"] < 0
    assert bool((o["member_logits"][dead] == -30).all())


def test_member_lists_threshold():
    lg = torch.tensor([[3.0, -1.0, 0.1], [-30.0, -30.0, 5.0]])
    pid = torch.tensor([7.0, 8.0, 9.0])
    assert member_lists(lg, pid, torch.tensor([True, True, True])) == [[7, 9], [9]]
    assert member_lists(lg, pid, torch.tensor([True, True, False])) == [[7], []]


# ---------------------------------------------------------------- data: tiling + dataset object ids
def _page(root, W=300, H=200):
    """One page image with 2 labelled objects (label lines 0 and 2; line 1 is another object) and its vector npz."""
    for d in ("images/train", "labels/train", "vectors/train", "images/val", "labels/val", "vectors/val"):
        (root / d).mkdir(parents=True, exist_ok=True)
    img = np.full((H, W, 3), 255, np.uint8)
    objs = [((40.0, 40.0), (50.0, 16.0), 0.0), ((150.0, 100.0), (30.0, 30.0), 0.0), ((250.0, 150.0), (40.0, 12.0), 0.0)]
    lines, rows = [], []
    for li, (c, wh, a) in enumerate(objs):
        q = cv2.boxPoints((c, wh, a))
        cv2.polylines(img, [q.astype(np.int32)], True, (0, 0, 0), 2)
        lines.append(f"{li % 2} " + " ".join(f"{v / (W if i % 2 == 0 else H):.6f}" for i, v in enumerate(q.reshape(-1))))
        for k in range(4):
            p = np.linspace(q[k], q[(k + 1) % 4], 8)
            rows.append(np.r_[0, p.reshape(-1), 0, 0, 0, 1.5, -1, li, 1000 + 10 * li + k])
    rows.append(np.r_[1, np.linspace([280, 10], [290, 20], 8).reshape(-1), 1, 0, 0, 1.0, -1, -1, 5])
    for split in ("train", "val"):
        cv2.imwrite(str(root / f"images/{split}/p0.png"), img)
        (root / f"labels/{split}/p0.txt").write_text("\n".join(lines) + "\n")
        np.savez_compressed(root / f"vectors/{split}/p0.npz", tokens=np.array(rows, np.float32))
    (root / "data.yaml").write_text("names: [a, b]\ntrain: images/train\nval: images/val\n")
    return objs


def test_tiling_keeps_inst_pid_and_maps_points(tmp_path):
    from openstroke.obb.data.prepare import prepare
    _page(tmp_path / "src")
    out = prepare(tmp_path / "src" / "data.yaml", tmp_path / "prep", size=128, gap=28, fit=False, workers=1)
    metas = [json.loads(l) for l in (out / "meta" / "train.jsonl").read_text().splitlines()]
    seen_pids = set()
    for m in metas:
        z = np.load(out / "vectors" / "train" / f"{m['name']}.npz")
        assert {"tokens", "inst", "pid", "view"} <= set(z.files) and len(z["inst"]) == len(z["pid"])
        assert m["n_vec"] == len(z["pid"])
        objs = m["objs"]
        for t, inst, pid in zip(z["tokens"], z["inst"], z["pid"]):
            pts = t[1:17].reshape(8, 2) * 128 + [m["x0"], m["y0"]]             # back to page pixels
            li = (pid - 1000) // 10 if pid >= 1000 else -1
            if pid >= 1000:                                                    # original points of that primitive
                seen_pids.add(int(pid))
                assert np.all(pts[:, 0] >= -1) and np.all(pts[:, 0] <= 301)
            if inst >= 0:                                                      # tile object index of the same object
                q = np.array(objs[inst][3:]).reshape(4, 2) + [m["x0"], m["y0"]]
                ctr = q.mean(0)
                assert li >= 0 and np.abs(pts.mean(0) - ctr).max() < 40           # its strokes are around its box
                assert objs[inst][0] == li % 2                                   # same class as the label line
        # every object of the tile owns the strokes of its label line that are in the tile
        for j, o in enumerate(objs):
            assert (z["inst"] == j).any()
    assert len(seen_pids) == 12


def test_dataset_inst_follows_augmentation(tmp_path):
    from openstroke.obb.data.prepare import prepare
    _page(tmp_path / "src")
    out = prepare(tmp_path / "src" / "data.yaml", tmp_path / "prep", size=128, gap=28, fit=False, workers=1)
    import random
    random.seed(0)
    np.random.seed(0)
    for kw in (dict(augment=False), dict(augment=True, fliplr=0.5, flipud=0.5),
               dict(augment=True, mosaic_p=1.0, mosaic_mode="yolo", scale_jitter=0.3, translate=0.1)):
        ds = DotaPatches(out, "train", size=128, hsv=(0, 0, 0), vectors=True, members=True, **kw)
        for i in range(len(ds)):
            _, t = ds[i]
            vec, boxes = t["vec"].numpy(), t["boxes"].numpy().copy()
            assert vec.shape[1] == 24
            boxes[:, :4] *= 128
            inst = np.rint(vec[:, 22]).astype(int)
            assert inst.max(initial=-1) < len(boxes)
            for j, b in enumerate(boxes):
                own = vec[inst == j]
                for row in own:                    # the object's strokes lie on its box (any augmentation)
                    p = row[1:17].reshape(8, 2) * 128
                    inside = [(abs((x - b[0]) * math.cos(b[4]) + (y - b[1]) * math.sin(b[4])) <= b[2] / 2 + 4) and
                              (abs(-(x - b[0]) * math.sin(b[4]) + (y - b[1]) * math.cos(b[4])) <= b[3] / 2 + 4)
                              for x, y in p if 0 <= x <= 128 and 0 <= y <= 128]
                    assert all(inside)


def test_token_cap_keeps_longest(tmp_path):
    from openstroke.obb.data.vectors import cap_order, cut_tile
    pts = np.zeros((5, 16), np.float32)
    for i, L in enumerate([1, 10, 3, 8, 2]):
        pts[i, 0::2] = np.linspace(0, L, 8)
    assert cap_order(pts, 2).tolist() == [1, 3]
    page = {"tokens": np.concatenate([np.zeros((5, 1)), pts + 5, np.zeros((5, 4)), -np.ones((5, 1)),
                                      -np.ones((5, 1)), np.arange(5)[:, None] + 50], 1).astype(np.float32)}
    st = {}
    t = cut_tile(page, 0, 0, 64, members=True, max_tokens=2, layers=False, stats=st)
    assert t.shape == (2, 24) and t[:, 23].tolist() == [51, 53] and (t[:, 22] == INST_UNKNOWN).all()
    assert st == {"tiles": 1, "tokens": 5, "capped": 1, "dropped": 3}
    t2 = cut_tile(page, 0, 0, 64, rate=0.5)                  # scaled image: points follow the rate
    assert np.allclose(t2[:, 1] * 64, 2.5)


def _ckpt(tmp_path, model):
    args = {"arch": "v25", "size": "s", "img": IMG, "num_top": 50, "groups": ",".join(map(str, G)), "vectors": True,
            "vec_dim": 32, "vec_layers": 1, "max_tokens": 8192, "member_thr": 0.5, "fit": False, "scale": 1.0,
            "nms_iou": 0.5}
    p = tmp_path / "v25.pt"
    torch.save({"model": model.state_dict(), "args": args, "classes": ["a", "b", "c"]}, p)
    return p


def test_predict_returns_members(tmp_path):
    from openstroke.obb.predict import run
    torch.manual_seed(0)
    m = OpenOBB25("s", num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1).eval()
    with torch.no_grad():                    # every refined box gets a high score: detections exist on a random net
        m.member.geo_w.copy_(torch.tensor([20.0, 0.0, 0.0, 0.0]))
    ck = _ckpt(tmp_path, m)
    _page(tmp_path / "src")
    res = run(str(ck), tmp_path / "src" / "images" / "val" / "p0.png", out=tmp_path / "pred", conf=0.0001,
              batch=4, device="cpu", verbose=False, tile=True)
    dets = next(iter(res.values()))
    assert dets and all("members" in d and all(isinstance(v, int) for v in d["members"]) for d in dets)
    pids = {1000, 1001, 1002, 1003, 1010, 1011, 1012, 1013, 1020, 1021, 1022, 1023, 5}
    assert all(set(d["members"]) <= pids for d in dets)
    saved = json.loads((tmp_path / "pred" / "p0.json").read_text())
    assert "members" in saved[0]
    from openstroke.obb.model import OpenOBB
    r = OpenOBB(str(ck)).predict(str(tmp_path / "src" / "images" / "val" / "p0.png"), conf=0.0001, verbose=False,
                                 device="cpu")[0]
    assert len(r.members) == len(r.obb.conf)


def test_registry():
    from openstroke.obb.cli import ARCH_KEYS, MODELS, RECIPE25
    from openstroke.obb.predict import KNOWN_ARCHS
    assert MODELS["openobb25x"] == ("v25", "x") and "v25" in KNOWN_ARCHS
    assert RECIPE25["vectors"] and RECIPE25["max_tokens"] == 8192 and RECIPE25["vec_ground"] == 1.0
    assert {"vectors", "vec_dim", "max_tokens"} <= set(ARCH_KEYS)


def _v24_ckpt(tmp_path):
    a, _ = _pair()
    args = {"arch": "v24", "size": "s", "img": IMG, "num_top": 50, "groups": "0,1,1", "lsk": True, "box": "dfl",
            "ref_k": 1500, "v7_off": "", "nms_iou": 0.5}
    p = tmp_path / "v24.pt"
    torch.save({"model": a.state_dict(), "args": args, "classes": ["x", "y", "z"]}, p)
    return p


def test_cli_builds_v25_from_a_v24_checkpoint(tmp_path, monkeypatch):
    """model=openobb25s weights=<v24 .pt>: OpenOBB25 recipe (vectors, 8192 tokens, membership weight 1), the v24
    switches carried over; member_weight= is an alias of vec_ground."""
    import openstroke.obb.train as trainer
    from openstroke.obb.cli import train
    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(" ".join(argv) + " "))
    _page(tmp_path / "src")
    ck = _v24_ckpt(tmp_path)
    train(str(tmp_path / "src" / "data.yaml"), model="openobb25s", weights=str(ck), epochs=1, batch=2, imgsz=128,
          tile=True, tile_scale=1.0, project=str(tmp_path / "runs"), cache_dir=str(tmp_path / "cache"), workers=1,
          member_weight=0.5)
    c = calls[-1]
    for flag in ("--arch v25 ", "--vectors ", "--max-tokens 8192 ", "--vec-ground 0.5 ", f"--weights {ck} ",
                 "--lsk ", "--box dfl ", "--ref-k 1500 "):
        assert flag in c, flag


@pytest.mark.skipif(os.environ.get("OPENOBB_SLOW") != "1", reason="CPU smoke train (set OPENOBB_SLOW=1)")
def test_train_smoke_from_v24(tmp_path):
    """One iteration of the real trainer: v25 from a v24 checkpoint on a tiled page with vectors, val with the member
    metric, best.pt round trip through predict with members."""
    from openstroke.obb.cli import train
    from openstroke.obb.predict import run
    _page(tmp_path / "src")
    ck = _v24_ckpt(tmp_path)
    r = train(str(tmp_path / "src" / "data.yaml"), model="openobb25s", weights=str(ck), epochs=1, batch=2, imgsz=128,
              tile=True, tile_scale=1.0, gap=28, project=str(tmp_path / "runs"), name="t", cache_dir=str(tmp_path / "cache"),
              workers=0, max_iters=1, threads=1, compile=False, channels_last=False, mosaic=0.0, vec_dim=32,
              vec_layers=1)
    assert Path(r.best).exists()
    a = torch.load(r.best, map_location="cpu", weights_only=False)["args"]
    assert a["arch"] == "v25" and a["vectors"] and a["max_tokens"] == 8192
    res = run(r.best, tmp_path / "src" / "images" / "val" / "p0.png", conf=0.0001, device="cpu", verbose=False,
              tile=True)
    assert all("members" in d for d in next(iter(res.values())))


def test_endpoint_geometry_distinguishes_crossing_and_preserves_direction_symmetry():
    from openstroke.obb.models.endpoint_members import endpoint_geometry

    boxes = torch.tensor([[64.0, 64.0, 60.0, 16.0, 0.0]])
    t = torch.linspace(-1, 1, 8)
    along = torch.stack([64 + 30 * t, torch.full_like(t, 56)], dim=-1)
    crossing = torch.stack([torch.full_like(t, 64), 64 + 12 * t], dim=-1)
    pts = torch.stack([along, crossing])
    rows, cols = torch.tensor([0, 0]), torch.tensor([0, 1])
    geom = endpoint_geometry(boxes, pts, rows, cols)
    assert torch.allclose(geom[0], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]))
    assert geom[1, 3] > 0 and geom[1, 5] == 0
    torch.testing.assert_close(geom, endpoint_geometry(boxes, pts.flip(1), rows, cols))
    rotated = torch.tensor([[64.0, 64.0, 16.0, 60.0, -math.pi / 2]])
    torch.testing.assert_close(geom, endpoint_geometry(rotated, pts, rows, cols))


def test_openstroke9_starts_as_v25_and_learns():
    torch.manual_seed(0)
    _, base = _pair()
    model = OpenOBB25("s", num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1,
                      endpoint_members=True).eval()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("member.endpoints.") for k in missing)
    x = torch.rand(2, 3, IMG, IMG)
    targets = _batch()
    ctx = ctx_batch(targets, torch.device("cpu"))
    with torch.no_grad():
        baseline, variant = base(x, ctx=ctx), model(x, ctx=ctx)
    for key in ("pred_boxes", "pred_logits", "member_logits"):
        torch.testing.assert_close(variant[key], baseline[key], rtol=0, atol=0)
    assert (variant["member_logits"] > -30).any()

    model.train()
    output = model(x, targets, ctx=ctx)
    loss = output["member_bce"] + output["member_dice"]
    assert loss.isfinite()
    loss.backward()
    assert model.member.endpoints.image.weight.grad.abs().sum() > 0
    assert model.member.endpoints.geometry.weight.grad.abs().sum() > 0


def test_openstroke9_cli_checkpoint_and_prediction(tmp_path, monkeypatch):
    import openstroke.obb.train as trainer
    from openstroke.obb.cli import train
    from openstroke.obb.predict import run
    from openstroke.obb.strip import strip

    model = OpenOBB25("s", num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1,
                      endpoint_members=True).eval()
    source = _ckpt(tmp_path, model)
    ck = torch.load(source, map_location="cpu", weights_only=False)
    ck["args"]["endpoint_members"] = True
    torch.save(ck, source)
    deploy = strip(str(source))
    data = tmp_path / "src"
    _page(data)
    result = run(str(deploy), data / "images" / "val" / "p0.png", out=tmp_path / "pred", conf=0.0001,
                 batch=4, device="cpu", verbose=False, tile=True)
    dets = next(iter(result.values()))
    assert dets and all("members" in d for d in dets)

    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(argv))
    for name, flags in (("inherited", {}), ("disabled", {"endpoint_members": False})):
        train(str(data / "data.yaml"), model=str(source), epochs=1, batch=2, imgsz=IMG, tile=True, tile_scale=1.0,
              project=str(tmp_path / "runs"), name=name, cache_dir=str(tmp_path / "cache"), workers=1, **flags)
    assert "--endpoint-members" in calls[0]
    assert "--endpoint-members" not in calls[1]
