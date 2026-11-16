"""Fit the face classifier of the HVAC post-processing (F151) and save it for the app (models/ is gitignored: no weights in Git).

    py -3.12 tools/fit_hvac_faces.py models/hvac_faces.joblib runs_colab/e114/faces_train.npz [more.npz ...] [--gt-only]

The npz come from research/probes/faces_train.py (faces of the GT strokes on the TRAIN pages, matched to the GT objects).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openstroke.hvac_faces import fit_classifier, save_classifier  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out, *rest = sys.argv[1:]
    npz = [a for a in rest if not a.startswith("--")]
    clf, classes = fit_classifier(npz, viewer="--gt-only" not in rest)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_classifier(out, clf, classes)
    print(f"saved {out}: classes {classes}, from {npz}")
