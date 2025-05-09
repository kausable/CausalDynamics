# Baseline

Here, we describe the baseline models and evaluation metrics used.

## Models
Examples to run baselines can be found in `notebooks/eval_pipeline.ipynb`. Follow installation and (more) runtime instructions of each baseline in the provided github links.


- [x] PCMCI+: https://github.com/jakobrunge/tigramite
- [x] FPCMCI: https://github.com/lcastri/fpcmci
- [x] VARLiNGAM: https://github.com/cdt15/lingam
- [x] DYNOTEARS: https://github.com/mckinsey/causalnex
- [x] Neural GC: https://github.com/iancovert/Neural-GC
- [x] CUTS+: https://github.com/jarrycyx/unn
- [x] TSCI: https://github.com/KurtButler/tangentspace

## Metrics
We use `AUROC` and `AUPRC` scores to evaluate the accuracy of discoverd summary graph.