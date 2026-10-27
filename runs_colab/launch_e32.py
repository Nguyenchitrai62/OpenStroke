import sys; sys.path.insert(0, 'tools'); import colab_run
D4 = "/content/dataset_local/e4"
C = "modes=fire,hvac feature_drop=layer_stats batch=4 lr0=0.002 image_branch=true img_lum=true restyle_mode=order"
colab_run.start("e32-g4", "datasets/openstroke_e4fh_colab.zip",
                f"put:datasets/e4/folds_pretrain_clean.json={D4}/folds_pretrain_clean.json",
                f"put:datasets/e3/folds_cv_pseudo.json={D4}/folds_cv_pseudo.json",
                f"openstroke train model=openstroke2 data={D4} folds={D4}/folds_pretrain_clean.json fold=0 {C} pretrain=0.4 epochs=40 tta=false name=e32p",
                f"openstroke train model=runs/e32p/fold0/best.pt data={D4} folds={D4}/folds_cv_pseudo.json fold=3,4 {C} layer_head=true epochs=15 name=e32")
