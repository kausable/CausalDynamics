# CausalDynamics: A large-scale benchmark for structural discovery of dynamical causal models


<div align="center">
<a href="http://kausable.github.io/CausalDynamics"><img src="https://img.shields.io/badge/View-Documentation-blue?style=for-the-badge)" alt="Homepage"/></a>
  <!-- <a href="<ADD_LINK>"><img src="https://img.shields.io/badge/ArXiV-2402.00712-b31b1b.svg" alt="arXiv"/></a> -->
<a href="https://huggingface.co/datasets/kausable/CausalDynamics"><img src="https://img.shields.io/badge/Dataset-HuggingFace-ffd21e" alt="Huggingface Dataset"/></a>
<a href="https://github.com/kausable/CausalDynamics/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/License-MIT-green" alt="License Badge"/></a>
<a href="https://github.com/kausable/CausalDynamics/actions/workflows/run-tests.yml"><img src="https://github.com/kausable/CausalDynamics/workflows/Run%20Tests/badge.svg" alt="Tests"/></a>
</div>
</br>

A comprehensive benchmark framework designed to rigorously evaluate state-of-the-art causal discovery algorithms for dynamical systems.

## Key Features
![Overview of CausalDynamics](docs/causaldynamics_overview.png)


1️⃣ **Large-Scale Benchmark**. Systematically evaluate state-of-the-art causal discovery algorithms on thousands of graph challenges with increasing difficulty. 

2️⃣ **Customizable Data Generation**. Scalable, user-friendly generation of increasingly complex coupled ordinary and stochastic systems of differential equations

3️⃣ **Diverse Challenges**. From simple chaotic systems to modular causal coupling of dynamical systems, including optional noise, confounding, time lags, and even climate model dynamics.

**Abstract**: Causal discovery for dynamical systems poses a major challenge in fields where active interventions are infeasible. Most methods used to investigate these systems and their associated benchmarks are tailored to deterministic, low-dimensional and weakly nonlinear time-series data. To address these limitations, we present *CausalDynamics*, a large-scale benchmark and extensible data generation framework to advance the structural discovery of dynamical causal models. Our benchmark consists of true causal graphs derived from thousands of coupled ordinary and stochastic differential equations as well as two idealized climate models. We perform a comprehensive evaluation of state-of-the-art causal discovery algorithms for graph reconstruction on systems with noisy, confounded, and lagged dynamics. *CausalDynamics* consists of a plug-and-play, build-your-own coupling workflow that enables the construction of a hierarchy of physical systems. We anticipate that our framework will facilitate the development of robust causal discovery algorithms that are broadly applicable across domains while addressing their unique challenges. 


## Installation

The easiest way to install the package is via PyPi:
```bash
pip install causaldynamics
```

Although you can generate your own dataset (see [getting started](#getting-started)), you can download our preprocessed ones directly from HuggingFace:
```bash
wget https://huggingface.co/datasets/kausable/CausalDynamics/resolve/main/process_causaldynamics.py
python process_causaldynamics.py
```

See the [additional installation guide](#additional-installation-guide) for more options.


## Getting Started

- [Challenge](https://kausable.github.io/CausalDynamics/challenge.html)
- [Quickstart](https://kausable.github.io/CausalDynamics/quickstart.html)
- [CausalDynamics](https://kausable.github.io/CausalDynamics/notebooks/causaldynamics.html)
- [Benchmark](https://kausable.github.io/CausalDynamics/benchmark.html)
    - [Simple Dynamics](https://kausable.github.io/CausalDynamics/notebooks/simple_causal_models.html)
    - [Coupled Dynamics](https://kausable.github.io/CausalDynamics/notebooks/coupled_causal_models.html)
    - [Climate Dynamics](https://kausable.github.io/CausalDynamics/notebooks/climate_causal_models.html)
- [Troubleshoot](https://kausable.github.io/CausalDynamics/troubleshoot.html)

## Benchmarking
- [Baseline](https://kausable.github.io/CausalDynamics/baseline.html)
- [Evaluation](https://kausable.github.io/CausalDynamics/notebooks/eval_pipeline.html)
- [Leaderboard](https://kausable.github.io/CausalDynamics/leaderboard.html)

## Additional Installation Guide
Note: This is the recommended way if you want to run scripts to generate benchmark data. Clone the repository and install it using [pdm](https://pdm-project.org/en/latest/): 

```shell
git clone https://github.com/kausable/CausalDynamics.git
pdm install
```

You can test whether the installation succeded by creating some coupled causal model data:

```shell
python src/causaldynamics/creator.py --config config.yaml
```

You find the output at `output/<timestamp>` as default location.

## Citation
If you find any of the code and dataset useful, feel free to acknowledge our work through:
