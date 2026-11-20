
import os

ROOT = os.environ.get("ESMEF_ROOT", ".")

DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
