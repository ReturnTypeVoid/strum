"""
STRUM: Spectral Transcription & Rhythm Understanding Model.

Convert songs (WAV/MP3) into playable charts for Clone Hero and YARG.
"""

__version__ = "0.1.0"
__author__ = "STRUM Team"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Default paths
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
