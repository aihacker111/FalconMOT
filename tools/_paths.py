"""Add the project root to ``sys.path`` so ``import falconmot`` works when a
tool is run directly (e.g. ``python tools/train.py``) without installing the
package. It walks up from this file until it finds the directory that contains
the ``falconmot`` package, so it keeps working from any sub-folder under
``tools/``. If FalconMOT is installed (``pip install -e .``) this is a no-op.
"""
import os.path as osp
import sys


def _find_repo_root(start: str) -> str:
    cur = osp.dirname(osp.abspath(start))
    while True:
        if osp.isdir(osp.join(cur, "falconmot")):
            return cur
        parent = osp.dirname(cur)
        if parent == cur:  # reached filesystem root
            return osp.dirname(osp.dirname(osp.abspath(start)))
        cur = parent


_ROOT = _find_repo_root(__file__)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
