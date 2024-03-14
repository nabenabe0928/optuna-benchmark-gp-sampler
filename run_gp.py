from __future__ import annotations

import json
from typing import Any

from hpolib import HPOLib

import numpy as np

import numpy as np

import optuna


cos30 = np.cos(np.pi / 6)
sin30 = np.sin(np.pi / 6)
DIM = 5
ROTATION = np.identity(DIM)
for i in range(DIM - 1):
    rotate = np.identity(DIM)
    rotate[i : i + 2, i : i + 2] = np.array([[cos30, -sin30], [cos30, sin30]])
    ROTATION = ROTATION @ rotate


def rotated_ellipsoid(trial: optuna.Trial) -> float:
    X = np.array([trial.suggest_float(f"x{i}", -5, 5) for i in range(DIM)])
    RX = ROTATION @ X
    weights = np.array([5**i for i in range(DIM)])
    return weights @ ((RX - 2) ** 2)


def optimize(n_trials: int, seed: int, sampler_name: str, deterministic: bool) -> list[optuna.trial.FrozenTrial]:
    assert sampler_name in ["gp", "botorch"]
    if sampler_name == "gp":
        sampler = optuna.samplers.GPSampler(seed=seed, deterministic_objective=deterministic)
    elif sampler_name == "botorch":
        sampler = optuna.integration.BoTorchSampler(seed=seed)
    else:
        assert False, f"Got an unknown name {sampler_name}."

    if deterministic:
        objective = rotated_ellipsoid
    else:
        objective = HPOLib(dataset_id=2, seed=seed)  # Naval

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.trials


def get_runtimes(trials: list[optuna.trial.FrozenTrial]) -> float:
    datetime_start = trials[0].datetime_start
    return [(t.datetime_complete - datetime_start).microseconds * 1e-6 for t in trials]


def get_values(trials: list[optuna.trial.FrozenTrial]) -> float:
    return [t.value for t in trials]


if __name__ == "__main__":
    n_trials = 200
    n_seeds = 10
    values = {"gp/disc": [], "gp/cont": [], "botorch/disc": [], "botorch/cont": []}
    runtimes = {"gp/disc": [], "gp/cont": [], "botorch/disc": [], "botorch/cont": []}
    for seed in range(n_seeds):
        for (name, det) in [("gp", True), ("gp", False), ("botorch", True), ("botorch", False)]:
            trials = optimize(n_trials=n_trials, seed=seed, sampler_name=name, deterministic=det)
            key = f"{name}/{'cont' if det else 'disc'}"
            values[key].append(get_values(trials))
            runtimes[key].append(get_runtimes(trials))

    with open("runtimes.json", mode="w") as f:
        json.dump(runtimes, f)
    with open("values.json", mode="w") as f:
        json.dump(values, f)
