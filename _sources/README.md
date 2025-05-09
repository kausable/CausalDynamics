# CausalDynamics: A large-scale benchmark for structural discovery of dynamical causal models


<div align="center">
<a href="http://kausable.github.io/CausalDynamics"><img src="https://img.shields.io/badge/View-Documentation-blue?style=for-the-badge)" alt="Homepage"/></a>
  <!-- <a href="<ADD_LINK>"><img src="https://img.shields.io/badge/ArXiV-2402.00712-b31b1b.svg" alt="arXiv"/></a> -->
<a href="https://huggingface.co/datasets/kausable/CausalDynamics"><img src="https://img.shields.io/badge/Dataset-HuggingFace-ffd21e" alt="Huggingface Dataset"/></a>
<a href="https://github.com/kausable/CausalDynamics/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/license-MIT-green" alt="License Badge"/></a>
</div>
</br>

Causal discovery for dynamical systems poses a major challenge in fields where active interventions are infeasible. However, most methods and their associated benchmarks are tailored to time-series data, which is often deterministic, low-dimensional, and weakly nonlinear. To address these limitations, we present *CausalDynamics*, a large-scale benchmark for advancing the structural discovery of dynamical causal models. The platform consists of true causal graphs with thousands of increasingly complex coupled ordinary and stochastic systems of differential equations. We perform comprehensive evaluation against state-of-the-art causal discovery algorithms on graph reconstruction in challenging yet realistic settings where the dynamics are noisy, confounded, and lagged. Finally, we extend our platform to include climate models derived from first principles. This enables a plug-and-play coupling workflow across component subsystems to construct a pseudo-realistic hierarchy of complexity. *CausalDynamics* will facilitate the development of robust causal discovery algorithms that are capable of handling diverse real-world chaotic systems. We provide a user-friendly platform, including a quickstart guide and documentation.

## Features
TODO: Short list of features and nice figures :) 

## Getting Started

- [Challenge](https://kausable.github.io/CausalDynamics/challenge.html)
- [Quickstart](https://kausable.github.io/CausalDynamics/quickstart.html)
- [Benchmark](https://kausable.github.io/CausalDynamics/benchmark.html)
- [Visual Feature Overview](https://kausable.github.io/CausalDynamics/notebooks/features.html)


## Installation

### Using pip
The easiest way to install the package is via pypi:
```bash
pip install causaldynamics
```

### Using pdm
Alternatively, if you want to install the repository locally, clone the repository and install it using [pdm](https://pdm-project.org/en/latest/):

```shell
pdm install
```

You can test whether the installation succeded by creating some coupled causal model data:

```shell
python src/causaldynamics/creator.py --config config.yaml
```

You find the output at `output/<timestamp>` as default location.

## Notebooks
You can find several notebook that explain the functionality of the repository. Good starting points are the `coupled_causal_models.ipynb` for details on how to generate coupled chaotic model time-series data and the `eval_pipeline.ipynb` for information on how the evaluation pipeline is performed.

## Scripts

We provide several scripts to generate benchmark data for simplex, coupled and climate models. have a look at `scripts/README.md` for more information.

## Baselines
Examples to run baselines can be found in `notebooks/eval_pipeline.ipynb`. Follow installation and (more) runtime instructions of each baseline in the provided github links.


- [x] PCMCI+: https://github.com/jakobrunge/tigramite
- [x] FPCMCI: https://github.com/lcastri/fpcmci
- [x] VARLiNGAM: https://github.com/cdt15/lingam
- [x] DYNOTEARS: https://github.com/mckinsey/causalnex
- [x] Neural GC: https://github.com/iancovert/Neural-GC
- [x] CUTS+: https://github.com/jarrycyx/unn
- [x] TSCI: https://github.com/KurtButler/tangentspace


## Troubleshooting
### Animations
To create animations as `.mp4` files, you need `ffmpeg` to be installed on your system:

```shell
brew install ffmpeg # for MacOS
apt install ffmpeg # for Linux
```
