# Quickstart

```shell
pdm install             # Install the package
$(pdm venv activate)    # Enter the virtual environment
```

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


We provide several scripts to generate benchmark data, see `scripts/README.md` for more information.

To create animations as `.mp4` files, you need `ffmpeg` to be installed on your system:

```shell
brew install ffmpeg # for MacOS
apt install ffmpeg # for Linux
```