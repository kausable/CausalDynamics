import importlib

# Baselines are imported lazily so that each one only requires its own
# optional dependencies, e.g. `from causaldynamics.baselines import CUTSPlus`
# works without tigramite, lingam, causal-learn or causalnex installed.
_BASELINES = {
    "TSCI": ".ccm",
    "CUTSPlus": ".cuts",
    "DYNOTEARS": ".dynotears",
    "NGC_LSTM": ".neuralgc",
    "FPCMCI": ".pcmci",
    "PCMCIPlus": ".pcmci",
    "VARLiNGAM": ".varlingam",
    "RCD": ".rcd",
    "GIN": ".gin",
    "GRASP": ".perm",
    "TCDF": ".tcdf",
}

__all__ = list(_BASELINES)


def __getattr__(name):
    if name not in _BASELINES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_BASELINES[name], __name__)
    return getattr(module, name)


def __dir__():
    return __all__
