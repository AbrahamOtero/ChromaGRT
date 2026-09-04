"""ChromaGRT training entry points."""

from src.training.model_backends.chromagrt_backend import predict_chromagrt, train_chromagrt

__all__ = ["predict_chromagrt", "train_chromagrt"]
