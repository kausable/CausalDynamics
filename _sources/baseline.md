# Baseline

Here, we describe the baseline models and evaluation metrics used.

## Models
Examples to run baselines can be found in `notebooks/eval_pipeline.ipynb`. Follow installation and (more) runtime instructions of each baseline in the provided github links.

- [x] PCMCI+: https://github.com/jakobrunge/tigramite
- [x] FPCMCI: https://github.com/lcastri/fpcmci
- [x] VARLiNGAM: https://github.com/cdt15/lingam
- [x] RCD: https://github.com/py-why/causal-learn
- [x] GIN: https://github.com/py-why/causal-learn
- [x] GRaSP: https://github.com/py-why/causal-learn
- [x] DYNOTEARS: https://github.com/mckinsey/causalnex
- [x] Neural GC: https://github.com/iancovert/Neural-GC
- [x] CUTS+: https://github.com/jarrycyx/unn
- [x] TSCI: https://github.com/KurtButler/tangentspace
- [x] TCDF: https://github.com/M-Nauta/TCDF

## Metrics
We use `AUROC` and `AUPRC` scores to evaluate the accuracy of discoverd summary graph. The structural Hamming distance (`SHD`) of the thresholded graph is reported as well.

## Installation
The baselines need optional dependencies. Install them with:

```bash
pip install "causaldynamics[baselines]"
# FPCMCI and PCMCI+ also need IDTxl, which is only available from GitHub
pip install cython
pip install --no-build-isolation git+https://github.com/pwollstadt/IDTxl.git
```

DYNOTEARS (`causalnex`) requires `pandas<2` and an outdated `numpy` pin, so install it in a separate environment:

```bash
pip install causaldynamics "pandas<2" "pgmpy<0.1.20" pathos pyvis
pip install --no-deps "causalnex @ git+https://github.com/mckinsey/causalnex.git@develop"
```

From a clone of the repository, use `uv sync --extra baselines --group idtxl`, or `uv sync --group causalnex` for DYNOTEARS. These install the exact versions from `uv.lock`.
