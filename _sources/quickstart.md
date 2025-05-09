# Quickstart

**Step 1**: Installation

- The easiest way to install the package is via pypi:
```bash
pip install causaldynamics
```

- Alternatively, if you want to install the repository locally, clone the repository and install it using [pdm](https://pdm-project.org/en/latest/):

```shell
git clone https://github.com/kausable/CausalDynamics.git
cd CausalDynamics
pdm install
```

**NOTE**: You can test whether the installation succeded by creating some coupled causal model data:

```shell
python src/causaldynamics/creator.py --config config.yaml
```