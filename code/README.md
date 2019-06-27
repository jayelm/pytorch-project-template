# code

This folder contains the main codebase, split into separate files, described as follows:

- `train.py`: Your main training script and entry point into the codebase.
- `io_util.py`: Utilities for (1) parsing arguments passed to various scripts (not just `train`, and not just from a CLI), (2) serialization of models and metrics.
- `util.py`: Other miscellaneous utilities.
- `models.py`: Implementations of the models that are being trained.
- `data.py`: Utilities for loading and preprocessing data from the `data/` folder in this repository.
- `builders.py`:
  - This particular file is quite important: We don't build our models, optimizers, or loss functions in `train.py`, because post-hoc analysis and testing scripts will also need to initialize such models (preferably from a serialized set of arguments). So we wrap all of this logic in `builders.py`. As much as possible, `builders.py` is designed to have functions that build models/losses given ONLY a set of arguments produced from `io_util.parse_args` for maximum reproducibility.

In general these scripts are expected to be run from the root repository directory, i.e. like `python code/train.py`.
