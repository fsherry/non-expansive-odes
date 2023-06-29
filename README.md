# Designing Stable Neural Networks using Convex Analysis and ODEs

This code repository accompanies the paper ["Designing Stable Neural Networks using Convex Analysis and ODEs"](https://arxiv.org/abs/2306.17332) by Ferdia Sherry, Elena Celledoni, Matthias J. Ehrhardt, Davide Murari, Brynjulf Owren and Carola-Bibiane Schönlieb.

The directory structure of the package is shown below:

    non_expansive_odes/
    ├── experiments
    ├── models
    └── utils
The directory `models` contains the code implementing our models and the models against which we have compared them, `utils` contains helper functionality, and `experiments` contains the actual scripts used to run the experiments in the paper.

## Installation and usage

To install, please use `poetry`:

    poetry install
The experiment scripts use `wandb` to log the training runs and assume that the environment variable `WANDB_DIR` is set. Set this using

    export WANDB_DIR=<path_to_wandb_directory>
before running the training scripts.
