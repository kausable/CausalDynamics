# CausalDynamics: A large-scale benchmark for structural discovery of dynamical causal models

CausalDynamics is a benchmark project designed to push the boundaries of causal discovery in complex dynamical systems.
It introduces a challenging suite of high-dimensional, noisy, and confounded dynamical environments.
These include thousands of coupled ordinary and stochastic differential equations, each grounded in known causal structures.
The benchmark evaluates the performance of state-of-the-art causal discovery methods under realistic conditions such as lagged dependencies and stochasticity that better reflect the nature of real-world systems.
Importantly, CausalDynamics supports a plug-and-play coupling workflow across component subsystems to construct a hierarchy of complexity.
The benchmark serves as a timely foundation for advancing the next generation of causal inference tools in scientific domains where dynamics are rich and interventions are rare.


## Installation
```shell
pdm install             # Install the package
$(pdm venv activate)    # Enter the virtual environment
```

## Usage
Simplest usage:
```shell
python src/causaldynamics/creator.py
```

You find the output at `output/<timestamp>` as default location.

There is a configuration file with all configurable parameters you can pass:
```shell
python src/causaldynamics/creator.py --config config.yaml
```

You can pass arguments directly through the CLI. Have a look at the options:
```shell
python src/causaldynamics/creator.py -h
```

### Scripts

We provide several scripts to generate benchmark data, see `scripts/README.md` for more information.

## Troubleshooting
### Animations
To create animations as `.mp4` files, you need `ffmpeg` to be installed on your system:

```shell
brew install ffmpeg # for MacOS
apt install ffmpeg # for Linux
```
## Baselines
Examples to run baselines can be found in `notebooks/eval_pipeline.ipynb`. Follow installation and (more) runtime instructions of each baseline in the provided github links.


- [x] PCMCI+: https://github.com/jakobrunge/tigramite
- [x] FPCMCI: https://github.com/lcastri/fpcmci
- [x] VARLiNGAM: https://github.com/cdt15/lingam
- [x] DYNOTEARS: https://github.com/mckinsey/causalnex
- [x] Neural GC: https://github.com/iancovert/Neural-GC
- [x] CUTS+: https://github.com/jarrycyx/unn
- [x] TSCI: https://github.com/KurtButler/tangentspace
- [ ] ...
