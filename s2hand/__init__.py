"""S2HAND model loading and inference utilities.

The heavyweight PyTorch loader is imported lazily so lightweight utilities such
as TailGuard can be used without constructing the model runtime.
"""

__all__ = ["NAMES", "S2Hand", "load_model", "prepare"]


def __getattr__(name):
    if name in __all__:
        from . import loader

        return getattr(loader, name)
    raise AttributeError(name)
