"""Add the project root to sys.path so `import falconmot` works when running
scripts directly (e.g. `python scripts/train.py`) without installing the package.
If FalconMOT is pip-installed (`pip install -e .`), this is a harmless no-op.
"""
import os.path as osp
import sys

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))   # repo root (contains falconmot/)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
