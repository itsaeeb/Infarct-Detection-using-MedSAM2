"""
predictor package
"""
try:
    from .predictor_wrapper import MedSAM2InfarctPredictor, DEFAULT_CONFIG_PATH
except (ImportError, ValueError):
    from predictor_wrapper import MedSAM2InfarctPredictor, DEFAULT_CONFIG_PATH

__all__ = [
    "MedSAM2InfarctPredictor",
    "DEFAULT_CONFIG_PATH"
]
