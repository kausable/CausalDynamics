# Baseline
We describe the baseline models and evaluation metrics.

## Models
We use the following baseline models. Implementation can be found in `src/causaldynamics/baselines/`

- [x] PCMCI+: https://github.com/jakobrunge/tigramite
- [x] FPCMCI: https://github.com/lcastri/fpcmci
- [x] VARLiNGAM: https://github.com/cdt15/lingam
- [x] DYNOTEARS: https://github.com/mckinsey/causalnex
- [x] Neural GC: https://github.com/iancovert/Neural-GC
- [x] CUTS+: https://github.com/jarrycyx/unn
- [x] TSCI: https://github.com/KurtButler/tangentspace

## Metrics
We use `AUROC` and `AUPRC` as our metrics to evaluate summary graph discovery.