from __future__ import annotations

import json
import os

import h5py

import numpy as np

import optuna


DATA_DIR_NAME = os.path.join(os.environ["HOME"], "hpo_benchmarks")
SEARCH_SPACE: dict[str, list[int] | list[float] | list[str]] = json.load(open("tabular_benchmarks.json"))


class HPOLib:
    """
    Download the datasets via:
        $ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
        $ tar xf fcnet_tabular_benchmarks.tar.gz
    """

    def __init__(self, dataset_id: int, seed: int | None):
        self.dataset_name = [
            "slice_localization",
            "protein_structure",
            "naval_propulsion",
            "parkinsons_telemonitoring",
        ][dataset_id]
        data_path = os.path.join(DATA_DIR_NAME, "hpolib", f"fcnet_{self.dataset_name}_data.hdf5")
        self._db = h5py.File(data_path, "r")
        self._rng = np.random.RandomState(seed)
        self._search_space = SEARCH_SPACE.copy()

    def reseed(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    def __call__(self, trial: optuna.Trial) -> float:
        seed = self._rng.randint(4)
        config: dict[int | str] = {}
        for param_name, possible_values in self._search_space.items():
            if isinstance(possible_values[0], str):
                config[param_name] = trial.suggest_categorical(param_name, possible_values.copy())
            else:
                n_grids = len(possible_values)
                config[param_name] = possible_values[trial.suggest_int(f"{param_name}_index", 0, n_grids - 1)]

        key = json.dumps(config, sort_keys=True)
        return np.log(self._db[key]["valid_mse"][seed][99])
