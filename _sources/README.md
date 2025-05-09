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
    - [Main Features](https://kausable.github.io/CausalDynamics/notebooks/features.html)
    - [Coupled Dynamics](https://kausable.github.io/CausalDynamics/notebooks/coupled_causal_models.html)
    - [Climate Dynamics](https://kausable.github.io/CausalDynamics/notebooks/climate_causal_models.html)
- [Troubleshoot](https://kausable.github.io/CausalDynamics/troubleshoot.html)

## Benchmarking
- [Baseline](https://kausable.github.io/CausalDynamics/baseline.html)
- [Evaluation](https://kausable.github.io/CausalDynamics/notebooks/eval_pipeline.html)
- [Leaderboard](https://kausable.github.io/CausalDynamics/leaderboard.html)

## Citation
If you find any of the code and dataset useful, feel free to acknowledge our work through:
