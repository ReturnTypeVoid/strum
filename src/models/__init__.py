"""Models module for instrument-specific neural networks."""

# Lazy imports to avoid loading tensorflow/tensorboard on startup
def __getattr__(name):
    if name == "DrumsCRNN":
        from src.models.drums import DrumsCRNN
        return DrumsCRNN
    if name == "GuitarCRNN":
        from src.models.guitar import GuitarCRNN
        return GuitarCRNN
    if name == "GuitarCRNN_Large":
        from src.models.guitar import GuitarCRNN_Large
        return GuitarCRNN_Large
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["DrumsCRNN", "GuitarCRNN", "GuitarCRNN_Large"]
