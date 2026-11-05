"""Dempster–Shafer fusion for match and time image pairs."""
from .config import load_config
from .pipeline import load_dataset, run_pair

__all__ = ["load_config", "load_dataset", "run_pair"]
