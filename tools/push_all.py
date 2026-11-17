"""Push this repo to GitHub, then sync the clean product subset to the company GitLab (HicasStroke), in one command (as OpenOBB).

    python tools/push_all.py              # git push (GitHub) + tag -> sync -> test -> push GitLab + tag
    python tools/push_all.py --dry-run    # show what would change on GitLab, push nothing
    python tools/push_all.py --no-test    # skip the test run on the GitLab copy (not recommended)
    python tools/push_all.py --minor      # bump 0.X.0 instead of the patch number (a new research architecture)

Versioning: openstroke/__init__.py `__version__` is the only version (pyproject reads it), 0.<latest architecture>.<patch> like OpenOBB.
When the package (openstroke/, pyproject.toml) changed since the commit that last set the version, the patch number is bumped and committed
first ("Version x.y.z"); every released version is tagged vX.Y.Z on GitHub and on GitLab.

GitHub (github.com/Nguyenchitrai62/OpenStroke) holds the code, research cards and Colab notebooks (Colab installs from it); the research
ledger (research/) stays on the research machine (user 2026-11-17). GitLab (openanything/hicasstroke) holds the product copy: ALLOW below
minus EXCLUDE (the OpenOBB-based detector of version B and its tests), with TRANSFORMS applied, renamed to HicasStroke. Its README.md comes
from tools/gitlab_README.md; its .gitignore is never overwritten. Only committed files are synced (`git archive HEAD`), commits go on top of
GitLab main (never a force-push), and nothing is pushed unless the GitLab copy passes its tests and contains every source file of the package.
If GitLab main is protected, the release goes to a branch release/vX.Y.Z for a merge request.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITLAB = "https://git.anybim.vn/CxDP/worker/modules/openanything/hicasstroke.git"     # product repo (user 2026-11-17)
GITLAB_PIP = "git.anybim.vn/CxDP/worker/modules/openanything/hicasstroke.git"
HICASOBB_PIP = "git.anybim.vn/CxDP/service/hicasai/openanything/hicasobb.git"
CHECKOUT = ROOT / ".gitlab_export"                     # persistent working clone of GitLab (gitignored)
ARCH_CARDS = [28, 29, 30]                              # research versions shipped as named architectures (models.ARCHS)
ALLOW = ["openstroke", "tests", *[f"docs/OPENSTROKE{n}.md" for n in ARCH_CARDS],
         "app/hvac_app.py", "app/hvac_viewer.html", "run_app.bat", "tools/extract_payload.py", "tools/run_hvac_logic.py",
         "pyproject.toml"]                              # internal company repo: no licence files (user 2026-11-17)
RETIRED = ["LICENSE", "LICENSES", "THIRD_PARTY_NOTICES.md"]   # paths of older layouts to remove from GitLab when present
README_SRC = "tools/gitlab_README.md"                  # GitLab README (product wording), copied as README.md
EXCLUDE_DIRS = ["openstroke/obb", "tests/obb"]          # version B: OpenOBB code (not a product part)
EXCLUDE: set[str] = set()
FORBIDDEN = re.compile(r"^\s*(import|from)\s+(ultralytics|fitz|pymupdf|mmcv|mmdet|mmrotate)\b", re.M)
# (file, pattern, replacement, is_regex): product wording for text that mentions research-only or excluded parts
TRANSFORMS = [
    ("openstroke/cli.py", r"    if args_in\[:1\] == \[\"obb\"\]:.*?return obb_main\(args_in\[1:\]\) or 0\n", "", True),
    ("pyproject.toml", r"(?m)^license(-files)? = .*\n", "", True),
    ("openstroke/pack.py", r"\(https://github\.com/Nguyenchitrai62/OpenStroke\)", f"(https://{GITLAB_PIP})", True),
]
# docs: Colab installs and links name the research repo; the product copy installs from GitLab
DOC_TRANSFORMS = [
    (r"git\+https://github\.com/Nguyenchitrai62/OpenStroke\.git(@main)?", f"git+https://{GITLAB_PIP}"),
    (r"https://github\.com/Nguyenchitrai62/OpenStroke(\.git)?", f"https://{GITLAB_PIP}"),
    (r"https://github\.com/Nguyenchitrai62/OpenOBB(\.git)?", f"https://{HICASOBB_PIP}"),
]

# GitLab branding (as HicasOBB): package openstroke -> hicasstroke, OpenStroke -> HicasStroke, cards OPENSTROKE<N> -> HICASSTROKE<N>,
# OpenOBB (the sibling detector) -> HicasOBB. Checkpoints hold only tensors / dicts / strings and are rebuilt from config["arch"]
# (openstroke28 / 29 / 30 -> hicasstroke28 / 29 / 30 are accepted by both names, see openstroke.models.ARCHS aliases).
PKG_NEW, CLASS_NEW = "hicasstroke", "HicasStroke"
KEEP = (GITLAB_PIP, HICASOBB_PIP)
TEXT_EXT = {".py", ".md", ".toml", ".txt", ".bat", ".cfg", ".ini", ".json", ".yaml", ".yml", ".html"}
TEXT_NAMES = {"LICENSE", "NOTICE", "COPYING"}


def is_text(f):
    return f.suffix.lower() in TEXT_EXT or f.name in TEXT_NAMES


# the product repo never names the research repo / package or the research ledger (checked before every push)
LEAK = re.compile(r"openstroke|openobb|nguyenchitrai|github\.com/nguyen|(?<!\w)research/|repo nghiên cứu|research repo", re.I)
SCRUB = (("research/LEDGER.md", "tài liệu nội bộ"), ("research/HYPOTHESES.md", "tài liệu nội bộ"), ("repo nghiên cứu", "nhóm phát triển"),
         ("research repository", "internal repository"), ("research repo", "internal repository"))


def rebrand_text(s):
    masks = {}
    for i, k in enumerate(KEEP):
        tok = f"\0KEEP{i}\0"
        s = s.replace(k, tok)
        masks[tok] = k
    s = s.replace("OpenOBB", "HicasOBB").replace("OPENOBB", "HICASOBB").replace("openobb", "hicasobb")
    s = s.replace("OpenStroke", CLASS_NEW).replace("OPENSTROKE", CLASS_NEW.upper()).replace("openstroke", PKG_NEW)
    s = re.sub(r"\[([^\]]*)\]\((?:\.\./)*research/[^)]*\)", r"\1", s)          # [text](../research/x) -> text
    for old, new in SCRUB:
        s = s.replace(old, new)
    s = re.sub(r"(?<!\w)research/[\w./-]*", "tài liệu nội bộ", s)
    for tok, k in masks.items():
        s = s.replace(tok, k)
    return s


def rebrand_path(rel):
    """Posix path relative to the repo -> its GitLab name."""
    return "/".join(rebrand_text(x) for x in rel.split("/"))


def leaks(dst):
    out = []
    for f in dst.rglob("*"):
        if f.is_file() and is_text(f):
            m = LEAK.search(f.read_text(encoding="utf-8"))
            if m:
                out.append((f.relative_to(dst).as_posix(), m.group(0)))
    return out


LOCAL_WHEELS = ROOT / "dist_hicasstroke"               # every released product wheel, kept locally (gitignored)


def build_wheel(src):
    """Product wheel of the exported tree -> LOCAL_WHEELS only (internal use; not committed to GitLab)."""
    import zipfile
    with tempfile.TemporaryDirectory() as tmp:
        run([sys.executable, "-m", "pip", "wheel", str(src), "--no-deps", "--no-build-isolation", "-q", "-w", tmp])
        run(["git", "clean", "-qfdX"], cwd=src)
        whl = next(Path(tmp).glob("*.whl"))
        with zipfile.ZipFile(whl) as z:
            bad = [n for n in z.namelist() if LEAK.search(n + z.read(n).decode("utf-8", "ignore"))]
        if bad:
            sys.exit(f"refusing to ship: research names inside the wheel: {bad[:5]}")
        LOCAL_WHEELS.mkdir(parents=True, exist_ok=True)
        shutil.copy(whl, LOCAL_WHEELS / whl.name)
    return LOCAL_WHEELS / whl.name


def rebrand(dst):
    files = [f for f in dst.rglob("*") if f.is_file()]
    for f in files:
        rel = f.relative_to(dst).as_posix()
        if is_text(f):
            f.write_text(rebrand_text(f.read_text(encoding="utf-8")), encoding="utf-8")
        new = dst / rebrand_path(rel)
        if new != f:
            new.parent.mkdir(parents=True, exist_ok=True)
            f.replace(new)
    for d in sorted((d for d in dst.rglob("*") if d.is_dir()), key=lambda d: -len(d.parts)):
        if not any(d.iterdir()):
            d.rmdir()


def run(cmd, cwd=ROOT, check=True, capture=False):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", OMP_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=capture, encoding="utf-8", errors="replace")
    if check and r.returncode:
        sys.exit(f"failed ({r.returncode}): {' '.join(map(str, cmd))}\n{(r.stdout or '')[-2000:]}{(r.stderr or '')[-2000:]}")
    return r


def build_tree(dst):
    """Committed ALLOW files of HEAD -> dst, minus EXCLUDE, with TRANSFORMS applied, renamed."""
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "a.tar"
        run(["git", "archive", "-o", str(tar), "HEAD", *ALLOW, README_SRC])
        with tarfile.open(tar) as t:
            t.extractall(dst, filter="data")
    for rel in EXCLUDE:
        (dst / rel).unlink(missing_ok=True)
    for rel in EXCLUDE_DIRS:
        shutil.rmtree(dst / rel, ignore_errors=True)
    for f in dst.rglob("__pycache__"):
        shutil.rmtree(f, ignore_errors=True)
    (dst / README_SRC).replace(dst / "README.md")
    for rel, pat, rep, is_re in TRANSFORMS:
        p = dst / rel
        s = p.read_text(encoding="utf-8")
        t = re.sub(pat, rep, s, flags=re.S if ".*?" in pat else 0) if is_re else s.replace(pat, rep)
        if t == s:
            print(f"  note: transform did not apply to {rel}: {pat[:60]!r}")
        p.write_text(t, encoding="utf-8")
    for f in [*(dst / "docs").glob("*.md"), dst / "README.md"]:
        s = f.read_text(encoding="utf-8")
        for pat, rep in DOC_TRANSFORMS:
            s = re.sub(pat, rep, s)
        f.write_text(s, encoding="utf-8")
    bad = [str(p.relative_to(dst)) for p in (dst / "openstroke").rglob("*.py") if FORBIDDEN.search(p.read_text(encoding="utf-8"))]
    if bad:
        sys.exit(f"refusing to sync: forbidden imports in {bad}")
    rebrand(dst)
    left = leaks(dst)
    if left:
        sys.exit(f"refusing to sync: research names left in the product copy: {left[:8]}")


VERSION_FILE = ROOT / "openstroke" / "__init__.py"
CODE_PATHS = ["openstroke", "pyproject.toml"]
VERSION_RE = re.compile(r'__version__ = "(\d+)\.(\d+)\.(\d+)"')


def read_version():
    m = VERSION_RE.search(VERSION_FILE.read_text(encoding="utf-8"))
    if not m:
        sys.exit("no __version__ = \"x.y.z\" in openstroke/__init__.py")
    return tuple(int(v) for v in m.groups())


def next_version(kind, dry_run):
    """Bump when package code changed since the commit that last set the version; returns the version to release."""
    cur = read_version()
    base = run(["git", "log", "-1", "--format=%H", "-G", '__version__ = "', "--", "openstroke/__init__.py"], capture=True).stdout.strip()
    changed = run(["git", "diff", "--name-only", f"{base}..HEAD", "--", *CODE_PATHS], capture=True).stdout.split() if base else ["(none)"]
    if not changed:
        return cur
    major, minor, patch = cur
    new = (major, minor + 1, 0) if kind == "minor" else (major, minor, patch + 1)
    tag = ".".join(map(str, new))
    print(f"== version {'.'.join(map(str, cur))} -> {tag} ({len(changed)} package files changed since the last release)")
    if not dry_run:
        text = VERSION_FILE.read_text(encoding="utf-8")
        VERSION_FILE.write_text(VERSION_RE.sub(f'__version__ = "{tag}"', text, count=1), encoding="utf-8")
        run(["git", "add", str(VERSION_FILE)])
        subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=ROOT, check=True, text=True,
                       input=f"Version {tag}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
    return new


def ensure_tag(tag, cwd):
    """Tag HEAD of `cwd` as `tag` and push it, unless the remote already has that tag."""
    if run(["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"], cwd=cwd, capture=True).stdout.strip():
        return
    run(["git", "tag", "-f", tag], cwd=cwd, capture=True)
    run(["git", "push", "-q", "origin", f"refs/tags/{tag}"], cwd=cwd)
    print(f"   tagged {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--minor", action="store_true", help="bump the minor version instead of the patch number")
    a = ap.parse_args()
    dirty = run(["git", "status", "--porcelain", "--", *ALLOW], capture=True).stdout.strip()
    if dirty:
        print("note: uncommitted changes are NOT synced:\n" + dirty)
    version = next_version("minor" if a.minor else "patch", a.dry_run)
    tag = "v" + ".".join(map(str, version))

    if not a.dry_run:
        print(f"== GitHub: git push ({tag})")
        run(["git", "push", "origin", "HEAD:main"])
        ensure_tag(tag, ROOT)

    print("== GitLab: update the working clone")
    if (CHECKOUT / ".git").exists():
        url = run(["git", "remote", "get-url", "origin"], cwd=CHECKOUT, capture=True).stdout.strip()
        if url != GITLAB:
            sys.exit(f"{CHECKOUT} is a clone of {url}, not {GITLAB}: remove it and run again")
        run(["git", "fetch", "-q", "--tags", "origin"], cwd=CHECKOUT)
        if run(["git", "rev-parse", "--verify", "-q", "origin/main"], cwd=CHECKOUT, check=False, capture=True).returncode == 0:
            run(["git", "checkout", "-q", "-B", "main", "origin/main"], cwd=CHECKOUT)
            run(["git", "reset", "-q", "--hard", "origin/main"], cwd=CHECKOUT)
        run(["git", "clean", "-qfdx"], cwd=CHECKOUT)
    else:
        run(["git", "clone", "-q", GITLAB, str(CHECKOUT)])
    if not (CHECKOUT / ".gitignore").exists():            # new project: anchored rules only
        (CHECKOUT / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\n/runs/\n/build/\n/dist/\n*.egg-info/\n/models/\n"
                                             "/datasets/\n*.pt\n", encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / "tree"
        new.mkdir()
        build_tree(new)
        managed = ALLOW + RETIRED + ["README.md"]
        for rel in managed + [rebrand_path(r) for r in managed]:     # replace the managed paths, keep .gitignore
            p = CHECKOUT / rel
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        shutil.copytree(new, CHECKOUT, dirs_exist_ok=True)
    run(["git", "add", "-A"], cwd=CHECKOUT)
    committed = {rebrand_path(f) for f in run(["git", "ls-files", "openstroke"], capture=True).stdout.split()
                 if not any(f.startswith(d + "/") for d in EXCLUDE_DIRS)}
    tracked = set(run(["git", "ls-files", PKG_NEW], cwd=CHECKOUT, capture=True).stdout.split())
    missing = sorted(committed - tracked)
    if missing:
        sys.exit(f"refusing to push: package files not tracked on GitLab (ignore rule?): {missing[:5]}")
    diff = run(["git", "diff", "--cached", "--stat"], cwd=CHECKOUT, capture=True).stdout.strip()
    if not diff:
        print("GitLab already up to date.")
        if not a.dry_run:
            ensure_tag(tag, CHECKOUT)
        return
    print(diff.splitlines()[-1] if diff else "")
    if a.dry_run:
        print(diff)
        print("(dry run: nothing pushed)")
        return
    if not a.no_test:
        print("== GitLab copy: pytest")
        r = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/"], cwd=CHECKOUT, check=False, capture=True)
        print((r.stdout or "").strip().splitlines()[-1] if r.stdout else (r.stderr or "")[-500:])
        if r.returncode:
            print((r.stdout or "")[-3000:])
            sys.exit("tests failed on the GitLab copy: nothing pushed to GitLab (GitHub already pushed)")
        run(["git", "clean", "-qfdX"], cwd=CHECKOUT)
    whl = build_wheel(CHECKOUT)
    print(f"   product wheel {whl.name} ({whl.stat().st_size / 1024:.0f} KB) kept in {LOCAL_WHEELS}")
    msg = f"chore(release): {tag}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"   # Conventional Commits (team rule)
    subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=CHECKOUT, input=msg, text=True, check=True)
    r = run(["git", "push", "-q", "origin", "HEAD:main"], cwd=CHECKOUT, check=False, capture=True)
    if r.returncode:                                       # protected main: hand the release over as a branch for a merge request
        branch = f"release/{tag}"
        print(f"   main refused ({(r.stderr or '').strip()[-200:]}); pushing {branch} for a merge request")
        run(["git", "push", "-q", "origin", f"HEAD:refs/heads/{branch}"], cwd=CHECKOUT)
        print(f"GitLab: branch {branch} pushed; open a merge request into main, then tag {tag} on the merge commit")
        return
    ensure_tag(tag, CHECKOUT)
    print("GitLab:", run(["git", "log", "--oneline", "-1"], cwd=CHECKOUT, capture=True).stdout.strip())


if __name__ == "__main__":
    main()
