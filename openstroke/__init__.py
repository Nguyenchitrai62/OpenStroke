"""OpenStroke: per-shape probabilities for the vector strokes of a PDF drawing page (raw output).

Input: the drawing commands of one page (payload JSON produced elsewhere, AI_Takeoff ``split_countfire``); output: one probability per
shape that it is a stroke to extract. What "a stroke to extract" means comes from the training data; building objects from strokes is
post-processing for earlier models. OpenStroke35 also predicts classes and piece membership. The package never imports PyMuPDF (AGPL).

Architectures are named by research version (``models.ARCHS``): openstroke28, openstroke29, openstroke30, openstroke31, openstroke34, openstroke35, openstroke36 (latest).
"""

__version__ = "0.36.2"   # 0.<latest research architecture>.<patch>, as OpenOBB; bumped by tools/push_all.py when the package changes
