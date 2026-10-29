"""S2HAND model loading and inference utilities.

The heavyweight PyTorch loader is imported lazily so lightweight utilities such
as TailGuard can be used without constructing the model runtime.
"""

__all__ = [
    "NAMES", "TRAINING_WARM_START", "S2Hand", "load_model",
    "load_training_checkpoint", "load_training_warm_start", "prepare",
    "prepare_training_warm_start",
]


def __getattr__(name):
    if name in __all__:
        from . import loader

        return getattr(loader, name)
    raise AttributeError(name)
