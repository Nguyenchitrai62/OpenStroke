"""OpenStroke32: region swap between train pages (copy-paste of GT objects for vector windows, H137)."""

import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

from openstroke.data import Augment, WindowDataset, collate, region_swap  # noqa: E402


def _win(n, x0, label_every, oinst_base, pos0=0, seed=0):
    """A synthetic window: n shapes on a 10 pt grid from x0; every label_every-th shape true, in objects of 2 shapes."""
    rng = np.random.default_rng(seed)
    xy = np.stack([x0 + 10.0 * (np.arange(n) % 10), 10.0 * (np.arange(n) // 10)], 1)
    lab = (np.arange(n) % label_every == 0).astype(np.uint8)
    oinst = np.where(lab > 0, oinst_base + np.cumsum(lab) // 2, -1).astype(np.int64)
    pos = pos0 + np.arange(n)
    edges = np.stack([pos[:-1], pos[1:]], 1)
    return {"x": rng.random((n, 47)).astype(np.float32), "xy": xy, "pts": np.zeros((n, 16, 2), np.float32),
            "layer": np.zeros(n, np.int64), "label": lab, "length": np.ones(n, np.float32), "ylayer": np.zeros(n, np.float32),
            "pos": pos, "lw": np.ones(n, np.float32), "mode": np.zeros(n, np.int64), "ltab": np.zeros((1, 8), np.int64),
            "style": np.arange(n) % 3, "edges": edges, "ekind": np.ones(len(edges), np.int64),
            "ocls": np.where(lab > 0, 1, 0).astype(np.int64), "octr": xy.copy(), "oinst": oinst}


def test_region_swap_replaces_a_square_by_a_donor_square_around_a_gt_shape():
    host = _win(100, 0.0, 5, 0)
    donor = _win(100, 1000.0, 3, 0, pos0=500, seed=1)
    out = None
    for seed in range(20):   # a draw that swaps something
        out = region_swap(host, donor, np.random.default_rng(seed))
        if len(out["x"]) != len(host["x"]) or not np.array_equal(out["pos"], host["pos"]):
            break
    n = len(out["x"])
    assert all(len(out[k]) == n for k in ("xy", "pts", "label", "length", "pos", "oinst", "ocls", "octr", "style", "lw"))
    new = out["pos"] > host["pos"].max()                 # donor shapes get fresh positions after the host's
    assert new.any() and len(np.unique(out["pos"])) == n
    assert (out["label"][new] > 0).any()                 # the donor square is centred on a true shape
    lo, hi = host["xy"].min(0), host["xy"].max(0)
    assert (out["xy"][new] >= lo - 1000).all() and (out["xy"][new][:, 0] < 900).all()   # moved from x >= 1000 into the host's area
    assert np.allclose(out["octr"][new], out["xy"][new])                                 # object centres move with their shapes
    ho = out["oinst"][~new]
    do = out["oinst"][new]
    assert not set(ho[ho >= 0]) & set(do[do >= 0])       # donor objects do not merge with host objects
    pos = set(out["pos"].tolist())
    assert all(a in pos and b in pos for a, b in out["edges"].tolist())   # links only between shapes still in the window
    assert len(out["edges"]) == len(out["ekind"])


def test_region_swap_respects_the_window_cap_and_skips_donors_without_gt():
    host = _win(100, 0.0, 5, 0)
    donor = _win(400, 1000.0, 1, 0, pos0=500, seed=1)
    for seed in range(10):
        out = region_swap(host, donor, np.random.default_rng(seed), side=(0.9, 0.9), cap=100)
        assert len(out["x"]) <= 100
    empty = _win(50, 1000.0, 10 ** 6, 0, pos0=500)
    empty["label"][:] = 0
    assert region_swap(host, empty, np.random.default_rng(0)) is host


def test_window_dataset_swaps_regions_and_batches(tmp_path):
    from test_model import _synthetic

    from openstroke.data import load_page

    data, manifest = _synthetic(tmp_path)
    stems = sorted(p.stem for p in data.glob("*.npz"))
    pages = [load_page(data, s, "hvac", tmp_path / "cache", graph=True) for s in stems[:2]]
    for p in pages:
        p.oinst = np.where(p.label > 0, np.arange(len(p)) // 4, -1).astype(np.int64)
        p.ocls = (p.label > 0).astype(np.int64)
        p.octr = p.xy.astype(np.float64).copy()
    on = WindowDataset(pages, 8, 128, Augment(paste=1.0, drop_tokens=0.0, fliplr=0, rot90=False, scale=0, color=0, gray=0), seed=3)
    off = WindowDataset(pages, 8, 128, Augment(paste=0.0, drop_tokens=0.0, fliplr=0, rot90=False, scale=0, color=0, gray=0), seed=3)
    swapped = sum(int(not np.array_equal(on[i]["pos"], off[i]["pos"])) for i in range(8))
    assert swapped >= 1
    assert all(len(on[i]["x"]) <= 128 for i in range(8))
    assert np.array_equal(on[2]["pos"], on[2]["pos"])     # reproducible from (seed, epoch, i)
    b = collate([on[i] for i in range(4)])
    assert b["x"].shape[0] == 4 and bool(b["emask"].any())


def test_training_with_region_swap(tmp_path):
    """Tiny CPU run through the real trainer with paste=1: it trains, logs the swap and the run's args keep it."""
    import json

    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from test_model import _synthetic

    from openstroke.cli import main
    from openstroke.folds import propose

    data, manifest = _synthetic(tmp_path)
    (data / "folds.json").write_text(json.dumps(propose(manifest, k=2)))
    common = [f"data={data}", "fold=0", "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu", "tta=false",
              f"project={tmp_path / 'runs'}"]
    assert main(["train", *common, "model=openstroke2", "dims=16,32", "enc_depths=1,1", "dec_depths=1", "patch=32", "epochs=1",
                 "paste=1.0", "name=v32"]) == 0
    run = tmp_path / "runs" / "v32"
    args = json.loads((run / "args.json").read_text(encoding="utf-8"))
    assert float(args["paste"]) == 1.0
    log = "".join(f.read_text(encoding="utf-8") for f in run.rglob("*.log"))
    assert "page mix 1.0 (region)" in log


def test_fg_swap_puts_another_pages_objects_on_this_pages_background():
    """H138 (user 2026-11-18): the host keeps only its OTHER shapes, the donor gives only its TRUE shapes (turned / mirrored / moved, links
    and object ids kept); distances between donor shapes survive the move."""
    from openstroke.data import fg_swap

    host = _win(100, 0.0, 5, 0)
    donor = _win(100, 1000.0, 3, 0, pos0=500, seed=1)
    donor["label"][1::3] = 1                                       # pairs of linked true shapes (links run i -> i + 1)
    donor["oinst"] = np.where(donor["label"] > 0, np.arange(100) // 3, -1)
    out = fg_swap(host, donor, np.random.default_rng(4))
    new = out["pos"] > host["pos"].max()
    assert (out["label"][~new] == 0).all() and (out["label"][new] > 0).all()          # host: background only, donor: objects only
    assert (~new).sum() == (host["label"] == 0).sum() and new.sum() == (donor["label"] > 0).sum()
    t = np.flatnonzero(donor["label"] > 0)
    a, b = donor["xy"][t], out["xy"][new]
    da = np.hypot(*(a[:, None] - a[None]).transpose(2, 0, 1))
    db = np.hypot(*(b[:, None] - b[None]).transpose(2, 0, 1))
    assert np.allclose(da, db)                                                           # a rigid move (turn / mirror / shift)
    lo, hi = host["xy"].min(0), host["xy"].max(0)
    c = (b.min(0) + b.max(0)) / 2
    assert (c >= lo - 1).all() and (c <= hi + 1).all()                                   # placed on the host's area
    assert np.allclose(out["octr"][new], out["xy"][new])
    pos = set(out["pos"].tolist())
    assert all(a_ in pos and b_ in pos for a_, b_ in out["edges"].tolist())
    donor_links = [(x, y) for x, y in out["edges"].tolist() if x > host["pos"].max() and y > host["pos"].max()]
    assert donor_links                                                                   # links between the donor's objects kept
    assert len(fg_swap(host, donor, np.random.default_rng(0), cap=50)["x"]) <= 50


def test_window_dataset_fg_and_both_modes(tmp_path):
    from test_model import _synthetic

    from openstroke.data import load_page

    data, manifest = _synthetic(tmp_path)
    stems = sorted(p.stem for p in data.glob("*.npz"))
    pages = [load_page(data, s, "hvac", tmp_path / "cache", graph=True) for s in stems[:2]]
    for mode in ("fg", "both"):
        ds = WindowDataset(pages, 6, 128, Augment(paste=1.0, paste_mode=mode, drop_tokens=0.0, fliplr=0, rot90=False, scale=0, color=0, gray=0),
                           seed=5)
        items = [ds[i] for i in range(6)]
        assert all(len(w["x"]) <= 128 for w in items)
        assert collate(items[:3])["x"].shape[0] == 3
