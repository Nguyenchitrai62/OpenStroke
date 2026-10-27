import sys, time; sys.path.insert(0, 'tools'); import colab_run, cx
VM, D4 = "e32-g4", "/content/dataset_local/e4"
C = ("modes=fire,hvac feature_drop=layer_stats batch=4 lr0=0.002 image_branch=true img_lum=true restyle_mode=order "
     "layer_head=true epochs=15")
while True:  # after E32's jobs (sequential is cheaper than two runs on one GPU, LEDGER F21)
    ok, out = cx.exec_py(VM, "import os;print('@@CX@@'+str(os.path.exists('/content/jobs.done')))", timeout=120)
    if "True" in out:
        break
    time.sleep(120)
colab_run.launch(VM, "pip install -q --force-reinstall --no-deps git+https://github.com/Nguyenchitrai62/OpenStroke.git",
                 f"openstroke train model=runs/e26p/fold0/best.pt data={D4} folds={D4}/folds_cv_pseudo.json fold=0,2 {C} "
                 "coincide=0.2 name=e36")
