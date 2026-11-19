# HVAC instance grouping on archived predictions

## Question and decision

E122 predicts whether linked strokes belong to one instance. Its original decoder joins every positive link. A path of positive links can therefore merge two objects even when a stronger negative link separates them. H142 keeps the predicted strokes, the scored pairs and the checkpoint fixed. The only change is how the decoder resolves contradictory links. GAT-CADNet motivates learning pairwise CAD instance relations. Mutex Watershed motivates processing attractive and repulsive evidence in descending order of confidence. Here both signs come from the same E122 logit on the existing CAD graph. No external code, weights or layers are needed for grouping. [GAT-CADNet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_GAT-CADNet_Graph_Attention_Network_for_Panoptic_Symbol_Spotting_in_CAD_CVPR_2022_paper.html) · [Mutex Watershed, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper.html)

Registered before replay: H142 passes with at least +0.020 micro F2 and no validation page losing more than 0.020 recall. Gain below +0.005 rejects it. Stroke threshold 0.75, edge boundary logit zero, priority by absolute logit, negative first on ties, then by sorted endpoint IDs. Every selected stroke including an isolate has a predicted instance. Sorting costs O(E log E); the decoder additionally maintains mutual exclusions between components. Runtime and memory on a live PDF have not been benchmarked.

## Data and protocol

Same E122 best epoch 7, TTA ×8, seed 0 and hvac10b fold as F191. Both decoders read the identical archived outputs. All five AI_Takeoff GT validation pages are scored. The replay checks the score-pack pseudo flag and the NPZ source, document SHA, page and stem. The historical training split contains 19 GT and 171 Viewer pages. This experiment fits nothing. It uses development validation previously consulted in model selection. There is no untouched test result and no GT-only training result here.

F191 matches predicted and GT instances one-to-one at stroke-length IoU ≥ 0.5, irrespective of HVAC class. Of 1,371 in-scope primary/reference objects, 1,324 own at least one positive-length labelled stroke. Main table uses these 1,324; the all-object sensitivity below counts the other 47 as misses. Layer-based pipeline results use another protocol. The historical E69 native 0.511 and E122 full post-processing 0.835 are separate references.

| GT val page | Original F2 / P / R | Signed F2 / P / R | Original → signed matches | Covered / raw GT |
|---|---|---|---:|---:|
| doc180 | .6770 / .5245 / .7301 | .7139 / .5309 / .7812 | 257 → 275 | 352 / 369 |
| doc187 | .8410 / .6946 / .8878 | .8621 / .6923 / .9184 | 348 → 360 | 392 / 404 |
| doc189 | .8616 / .6884 / .9195 | .8584 / .6782 / .9195 | 137 → 137 | 149 / 157 |
| doc200 | .8101 / .6170 / .8788 | .8448 / .6150 / .9318 | 116 → 123 | 132 / 132 |
| doc201 | .9041 / .8040 / .9331 | .9220 / .8056 / .9565 | 279 → 286 | 299 / 309 |
| **Micro** | **.8097 / .6591 / .8588** | **.8327 / .6579 / .8920** | **1,137 → 1,181** | **1,324 / 1,371** |

H142 passes on E122. The signed decoder predicts 1,795 groups, up from 1,725. It adds 44 matches and 26 unmatched predictions. Doc189 loses 0.0032 F2. Recomputing micro F2 after omitting each PDF in turn gives gains from +.0182 to +.0264. This describes dependence on these five PDFs; independent test performance is unknown. Counting all 1,371 GT objects yields F2 / P / R **.7886 / .6591 / .8293 → .8112 / .6579 / .8614**. Both variants still match without class. Only 24 of the 1,324 represented objects have less than half their true stroke length selected; for the other misses, the selected length alone cannot explain failure.

| True class (e11 taxonomy) | Original matches | Signed matches | Covered / raw | Original → signed recall on covered |
|---|---:|---:|---:|---:|
| Duct | 547 | 557 | 642 / 677 | .852 → .868 |
| Elbow | 182 | 205 | 217 / 217 | .839 → .945 |
| Square elbow | 12 | 12 | 18 / 19 | .667 → .667 |
| Reducer (merged e11 class) | 160 | 163 | 183 / 189 | .874 → .891 |
| Branch | 225 | 233 | 250 / 255 | .900 → .932 |
| Tee | 11 | 11 | 14 / 14 | .786 → .786 |

The E122 stroke head supplies membership only. Predicted class precision and F2 cannot be computed from it. E127b's 0.563 on pieces is a **different, class-aware** validation metric with a different representation and denominator.

Secondary within-checkpoint replay on E120 (e10, TTA ×1, stroke threshold 0.70): 19 GT train pages .5973/.3232/.7579 → .6224/.3311/.7979 F2/P/R; five GT val pages .7168/.3948/.9003 → .7178/.3918/.9063. Transfer of the grouping gain to E120 validation is negligible. The local E122 all_scores archive covers only its five GT validation pages; no E122 train/val object gap was measured. A perfect accept/reject decision on original E122 components, keeping membership fixed, tops out at .8837 on covered validation GT (+.0740). This falls below the +.10 gate for designing a verification head on that evidence.

## Reproduction and inference

Repository revision at the start of the replay: `5b9fe05`. e11 manifest SHA256 `6773354e1a741078288937c8460c7f836b05062e3b95e3df163cb01f2ae59c81`, folds `ec98ca2c45d93f142e4e1b2dfb4be3894537a7dc035c847a04db11caba084598`, E122 val_scores `e5a72afd451106037d8c5efcadb5965ffb6672bbaa4f3611656a2d553c0c2daa`.

```sh
python tools/replay_stroke_groups.py datasets/e11 datasets/score_pack_e8_all runs_colab/e122/fold0/val_scores.npz --threshold 0.75
python tools/replay_stroke_groups.py datasets/e10 datasets/score_pack_e8_all runs_colab/e120/fold0/gt_tta1.npz --threshold 0.70 --split train
python tools/replay_stroke_groups.py datasets/e10 datasets/score_pack_e8_all runs_colab/e120/fold0/gt_tta1.npz --threshold 0.70 --split val
```

```python
from openstroke.infer import Predictor

predictor = Predictor(checkpoint)
out = predictor.predict_payload(payload, mode="hvac", tta=predictor.tta, group_strokes=True)
instance_id = out["instance_id"]  # one id per original payload shape; -1 below checkpoint threshold
```

## H143 on E127b, registered before execution

Use the existing score archive and val candidates from E122 picks. Compare separately (1) signed vs original links with the same predicted-class restriction and (2) selecting pieces by parent probability vs their own probability at 0.5 with original links. All arms use identical archived class outputs and length-majority class matching. Each arm passes at +0.020 micro class-aware F2 with worst per-page recall loss ≤ 0.020; below +0.005 rejects. Replay reports per-page and per-class F2/P/R and edge precision/recall by relation. Per-class denominators count the union of item and piece GT IDs. To compare to F194 the aggregate retains its historical maximum-of-counts convention. Touch/facing link diagnostics are conditional on two true pipe pieces.

Colab with Drive mounted (the E127b archive was not present in this workspace):

```python
!pip install -q --no-deps --force-reinstall "git+https://github.com/Nguyenchitrai62/OpenStroke.git@main"
!mkdir -p /content/os34 && unzip -qo /content/drive/MyDrive/Dataset/openstroke_e12_data.zip -d /content/os34
!python -m openstroke.replay_pieces /content/os34/e12 /content/drive/MyDrive/Train/OpenStroke34/e127b_pieces_e122/fold0/val_scores.npz
```

The replay checks the original 3,025 predictions, 1,368 GT and rounded F2 of .563 before declaring either arm a success. Class-aware results for both alternatives are **unmeasured**. Only a positive result there warrants a prospective GT-only, PDF-grouped evaluation with its own baseline and held-out test before a product accuracy claim.
