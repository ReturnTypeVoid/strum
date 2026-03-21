"""Inference module for generating charts from audio."""

__all__ = []

try:
    from src.inference.drums import infer_drums
    __all__.append("infer_drums")
except ImportError:
    pass

try:
    from src.inference.pipeline import run_inference, run_batch_inference
    __all__.extend(["run_inference", "run_batch_inference"])
except ImportError:
    pass

try:
    from src.inference.guitar_bass import transcribe_guitar, transcribe_bass
    __all__.extend(["transcribe_guitar", "transcribe_bass"])
except ImportError:
    pass
