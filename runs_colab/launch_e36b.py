import sys, glob
from pathlib import Path
sys.path.insert(0, 'tools'); import colab_run
D4 = "/content/dataset_local/e4"
C = ("modes=fire,hvac feature_drop=layer_stats batch=4 lr0=0.002 image_branch=true img_lum=true restyle_mode=order "
     "layer_head=true epochs=15")
puts = [f"put:{f}=/content/runs/{Path(f).relative_to('runs_colab').as_posix()}"
        for f in sorted(glob.glob('runs_colab/e32/**/*', recursive=True)) + ['runs_colab/e32p/fold0/best.pt',
                                                                            'runs_colab/e26p/fold0/best.pt'] if Path(f).is_file()]
puts += [f"put:datasets/e3/folds_cv_pseudo.json={D4}/folds_cv_pseudo.json"]
colab_run.start("e36-g4", "datasets/openstroke_e4fh_colab.zip", *puts,
                f"openstroke train model=runs/e26p/fold0/best.pt data={D4} folds={D4}/folds_cv_pseudo.json fold=0,2 {C} coincide=0.2 name=e36",
                "openstroke train resume=runs/e32")
