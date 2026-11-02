# Third-party notices

OpenStroke (this repository) is proprietary software of the repository owner (see `LICENSE`).
It incorporates a small amount of third-party material under permissive licenses. Those licenses allow commercial use,
modification and closed-source distribution, provided the copyright notices and the license text are kept.

| Component | Upstream | License | Files in this repo | Changes |
|---|---|---|---|---|
| Lovász extension gradient (binary Lovász hinge / Lovász-Softmax loss) | Berman et al., CVPR 2018, github.com/bermanmaxim/LovaszSoftmax | MIT | `openstroke/models/openstroke2.py` (`lovasz_hinge`, `_lovasz_grad`, `lovasz_sigmoid`) | written for flat binary token sets; the gradient follows the reference implementation closely, so it is treated as derived |
| HGNetv2 backbone | D-FINE (github.com/Peterande/D-FINE), ported from PaddleDetection, through OpenOBB | Apache-2.0 | `openstroke/models/hgnetv2.py` | config registry and distributed coupling removed; FrozenBatchNorm2d from torchvision |
| ResNet18 (code loaded at run time, not copied) and its ImageNet weights `ResNet18_Weights.IMAGENET1K_V1` | torchvision, github.com/pytorch/vision | BSD-3-Clause (code); weights: see below | `openstroke/models/image_branch.py` | truncated and extended with an FPN by OpenStroke |

License texts: `LICENSES/MIT-LovaszSoftmax.txt`, `LICENSES/BSD-3-Clause-torchvision.txt`, `LICENSES/Apache-2.0-D-FINE.txt`. This file and `LICENSES/` are shipped inside the wheel
(`dist-info/licenses/`).

All other code in `openstroke/`, `app/`, `tools/`, `tests/` and `research/` was written for OpenStroke. `tools/cx.py` and the training framework
follow OpenOBB, a project of the same owner. Unlike OpenOBB, this repository has **not** had a token-level comparison against upstream code.

## Pretrained weights

- `image_branch=true` (every best configuration since E28) initialises from torchvision's **ImageNet** ResNet18 weights, so every
  checkpoint trained with the image branch derives from them. torchvision's documentation says pretrained weights may have their own licenses or
  terms and leaves it to the user to check them; ImageNet itself is distributed for non-commercial research. **Check this before distributing
  commercial weights.** `img_pretrained=false` trains without them, but no result in this repository was measured that way.
- `img_backbone=hgnetv2_b5` initialises from a backbone trained by OpenOBB on the owner's Wall_Color floorplans (lineage D-FINE COCO, Apache-2.0).
- Weights are not in this repository. Training data comes from the owner's AI_Takeoff database and is not distributed either.

## Dependencies (installed by pip, not distributed)

| Package | License | Used for |
|---|---|---|
| PyTorch | BSD-style (mixed, see the package) | model, training |
| torchvision | BSD-3-Clause | ResNet18 of the image branch |
| NumPy, SciPy | BSD-3-Clause (mixed, see the packages) | features, graph |
| Shapely | BSD-3-Clause | labelling geometry |
| Matplotlib | Matplotlib license (PSF-style) | `results.png` |
| Streamlit (`[app]`) | Apache-2.0 | test app |
| scikit-learn (`[baseline]`) | BSD-3-Clause | `baseline-gbdt` |

## Not part of OpenStroke

- **PyMuPDF (AGPL-3.0, or a paid Artifex licence)** is what AI_Takeoff's `pdf_etl` uses to read PDF vectors. The `openstroke` package never imports
  it (enforced by `tests/test_e0.py`, `test_package_does_not_import_pymupdf`) and only reads the payload JSON. `tools/extract_payload.py`,
  `tools/make_payloads.py` and `tools/make_unlabeled.py` do import it, and run only under AI_Takeoff's own Python as separate processes; they are not
  in the wheel. The test app needs such a Python to open a PDF. A product distributed without an AGPL obligation needs another PDF reader, or the
  commercial PyMuPDF licence.
- **Ultralytics (AGPL-3.0)** is not used anywhere in this repository.
- Ideas taken from published papers (Point Transformer V3, LitePT, RoFormer, SymPoint-V2, VecFormer, CADTransformer, GAT-CADNet, Sonata, FPN, ...) are
  re-implemented here (see the module docstrings); this has not been checked by a token-level comparison.
