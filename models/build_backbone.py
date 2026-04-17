import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if DUST3R_REPO_ROOT not in sys.path:
    sys.path.insert(0, DUST3R_REPO_ROOT)

from dust3r.model import load_model


def build_dust3r_backbone(weights_path, device="cuda"):
    backbone = load_model(weights_path, device=device, verbose=True)
    backbone.train()
    return backbone