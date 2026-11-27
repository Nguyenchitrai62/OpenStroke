"""Export the HVAC duct post-processing into the extract-duct module as standalone code (user 2026-11-27).

The extract-duct module must not import the model package (the model research repo stays private): this script copies
only the functions the post-processing really runs (tree-shaken by AST from the entry points below), rewrites the
relative imports between the copied modules, drops every comment and docstring (``ast.unparse``) and refuses to write
when a research / package name is left. Re-run after a post-processing change, then run the module tests:

    py -3.12 tools/export_duct_post.py F:/Source_code/_AI_system/Module/extract_duct/src/extract_duct/hvac/vector_post

The parity check (``--check <page.json[.gz]>``) runs the original and the exported code on one page and compares the
objects exactly.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PKG = ROOT / "openstroke"

#: source module -> exported module name
MODULES = {"geometry": "_geometry", "labels": "_labels", "build": "_attributes", "objects": "_objects",
           "hvac_faces": "_faces"}
#: what the module calls
ENTRY = {
    "geometry": ["page_segments", "PageSegments"],
    "build": ["shape_attributes"],
    "hvac_faces": ["faces_first_hybrid", "hybrid_group_objects", "segment_objects", "load_classifier"],
}
FORBIDDEN = re.compile(r"openstroke|hicasstroke|openobb|hicasobb|ai_takeoff|github|anybim", re.IGNORECASE)
HEADER = ('"""Generated: HVAC duct post-processing ({name}). Do not edit by hand; regenerate it from the model team."""\n'
          "# ruff: noqa\n# fmt: off\n")


def top_level(tree: ast.Module) -> dict[str, list[ast.stmt]]:
    """name -> top-level statements that bind it (functions, classes, assignments, imports)."""
    out: dict[str, list[ast.stmt]] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                names += [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [(a.asname or a.name).split(".")[0] for a in node.names]
        for n in names:
            out.setdefault(n, []).append(node)
    return out


def references(node: ast.AST) -> tuple[set[str], list[tuple[str, str]]]:
    """Global-ish names a statement reads, and the (module, name) it imports relatively inside its body."""
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    rel = []
    for n in ast.walk(node):
        if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module:
            rel += [(n.module, a.name) for a in n.names]
    return names, rel


def shake() -> dict[str, list[ast.stmt]]:
    trees = {m: ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8")) for m in MODULES}
    tables = {m: top_level(t) for m, t in trees.items()}
    keep: dict[str, set[int]] = {m: set() for m in MODULES}
    used: dict[int, set[str]] = {}   # top-level relative import -> the names really needed from it
    todo = [(m, n) for m, names in ENTRY.items() for n in names]
    seen: set[tuple[str, str]] = set()
    while todo:
        mod, name = todo.pop()
        if (mod, name) in seen:
            continue
        seen.add((mod, name))
        if mod not in MODULES:
            raise SystemExit(f"{name} needs module {mod!r}, which is not exported: add it to MODULES")
        for node in tables[mod].get(name, []):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                keep[mod].add(id(node))
                used.setdefault(id(node), set()).add(name)
                todo += [(node.module, a.name) for a in node.names if (a.asname or a.name) == name]
                continue
            if id(node) in keep[mod]:
                continue
            keep[mod].add(id(node))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names, rel = references(node)
            todo += [(mod, n) for n in names if n in tables[mod]]
            todo += rel
    for tree in trees.values():
        for node in tree.body:
            if id(node) in used:
                node.names = [a for a in node.names if (a.asname or a.name) in used[id(node)]]
    return {m: [n for n in trees[m].body if id(n) in keep[m]] for m in MODULES}


class Rewrite(ast.NodeTransformer):
    """Relative imports -> the exported names; docstrings out."""

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level == 1:
            if node.module not in MODULES:
                raise SystemExit(f"exported code imports .{node.module}: export it too or cut the dependency")
            node.module = MODULES[node.module]
        return node

    def _strip_doc(self, node):
        self.generic_visit(node)
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _strip_doc


def render(mod: str, nodes: list[ast.stmt]) -> str:
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)]
                        + [n for n in nodes if not (isinstance(n, ast.ImportFrom) and n.module == "__future__")],
                        type_ignores=[])
    module = ast.fix_missing_locations(Rewrite().visit(module))
    code = HEADER.format(name=MODULES[mod].lstrip("_")) + ast.unparse(module) + "\n"
    return code


def export(out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for stale in out.glob("_*.py"):
        stale.unlink()
    for mod, nodes in shake().items():
        if not nodes:   # nothing of this module is used
            continue
        code = render(mod, nodes)
        bad = sorted({m.group(0) for m in FORBIDDEN.finditer(code)})
        if bad:
            lines = [ln for ln in code.splitlines() if FORBIDDEN.search(ln)][:5]
            raise SystemExit(f"{mod}: exported code still names {bad}:\n" + "\n".join(lines))
        compile(code, MODULES[mod], "exec")
        (out / f"{MODULES[mod]}.py").write_text(code, encoding="utf-8")
        files[f"{MODULES[mod]}.py"] = hashlib.sha256(code.encode()).hexdigest()
    init = out / "__init__.py"
    init.write_text('"""Generated HVAC duct post-processing (see the module README). Do not edit by hand."""\n\n'
                    "from ._attributes import shape_attributes\n"
                    "from ._faces import faces_first_hybrid, hybrid_group_objects, load_classifier, segment_objects\n"
                    "from ._geometry import PageSegments, page_segments\n\n"
                    '__all__ = ["PageSegments", "faces_first_hybrid", "hybrid_group_objects", "load_classifier", '
                    '"page_segments", "segment_objects", "shape_attributes"]\n', encoding="utf-8")
    from openstroke import __version__

    (out / "SOURCE.json").write_text(json.dumps({"version": __version__, "files": files}, indent=2) + "\n", encoding="utf-8")
    return files


def check(out: Path, page: Path, faces_model: Path) -> None:
    """Original vs exported code on one real page: identical objects / classes / instances."""
    import gzip
    import importlib.util

    import numpy as np

    from openstroke import hvac_faces as hf
    from openstroke.build import shape_attributes
    from openstroke.geometry import page_segments

    spec = importlib.util.spec_from_file_location("vector_post", out / "__init__.py", submodule_search_locations=[str(out)])
    vp = importlib.util.module_from_spec(spec)
    sys.modules["vector_post"] = vp
    spec.loader.exec_module(vp)

    opener = gzip.open if page.suffix == ".gz" else open
    shapes = json.load(opener(page, "rt", encoding="utf-8"))["shapes"]
    rng = np.random.default_rng(0)
    n = len(shapes)
    prob = rng.random(n)
    groups = {"instance_id": np.where(prob > 0.85, rng.integers(0, 80, n), -1), "cluster_ids": np.arange(80),
              "cluster_probs": rng.dirichlet(np.ones(8), 80)}
    results = []
    for lib, segf, attf in ((hf, page_segments, shape_attributes), (vp, vp.page_segments, vp.shape_attributes)):
        segs = segf(shapes)
        attrs = attf(shapes, segs)
        clf = lib.load_classifier(str(faces_model))
        o1, c1, i1 = lib.faces_first_hybrid(segs, prob, np.asarray(attrs["op_counts"])[:, 1] > 0, segs.shape_length(), *clf,
                                            groups, thr=0.85, attrs=attrs)
        o2, c2, i2 = lib.hybrid_group_objects(segs, prob, segs.shape_length(), groups, thr=0.85)
        s1 = lib.segment_objects(o1, segs, prob >= 0.85)
        results.append((json.dumps(o1, sort_keys=True, default=str), c1.tobytes(), i1.tobytes(),
                        json.dumps(o2, sort_keys=True, default=str), c2.tobytes(), i2.tobytes(), np.asarray(s1).tobytes(),
                        len(o1), len(o2)))
    assert results[0] == results[1], "exported post-processing differs from the original"
    print(f"parity OK on {page.name}: {results[0][7]} faces-hybrid objects, {results[0][8]} end-to-end objects")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--check", type=Path, help="page JSON (.json / .json.gz) for the parity check")
    ap.add_argument("--faces-model", type=Path, default=ROOT / "models" / "hvac_faces.joblib")
    a = ap.parse_args()
    files = export(a.out)
    print(f"exported {len(files)} files to {a.out}: " + ", ".join(files))
    if a.check:
        check(a.out, a.check, a.faces_model)


if __name__ == "__main__":
    main()
