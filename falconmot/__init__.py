"""FalconMOT — Multi-Object Tracking with DINOv3STAs + HybridEncoder + DEIMTransformer."""

__version__ = "0.2.0"

from .nn import create_model, load_model, save_model
from .cfg.args import opts
from .utils.log import Logger, get_logger
