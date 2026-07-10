import importlib

import numpy as np
import numpy.testing as npt
import pytest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import random as backend_random

simulation_module = importlib.import_module(
    "pyrecest.evaluation.generate_simulated_scenarios"
)


def _assert_random_states_equal(actual, expected):
    if isinstance(expected, tuple):
        assert actual[0] == expected[0]
        npt.assert_array_equal(actual[1], expected[1])
        assert actual[2:] == expected[2:]
        return

    if hasattr(actual, "detach"):
        actual = actual.detach().cpu().numpy()
    if hasattr(expected, "detach"):
        expected = expected.detach().cpu().numpy()
    npt.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _patch_minimal_simulation(monkeypatch, *, fail=False):
    monkeypatch.setattr(
        simulation_module,
        "check_and_fix_config",
        lambda simulation_params: simulation_params,
    )

    def generate_groundtruth(simulation_params):
        del simulation_params
        np.random.random()
        if fail:
            raise RuntimeError("generation failed")
        return np.array([0.0])

    monkeypatch.setattr(
        simulation_module,
        "generate_groundtruth",
        generate_groundtruth,
    )
    monkeypatch.setattr(
        simulation_module,
        "generate_measurements",
        lambda groundtruth, simulation_params: np.array(
            [float(groundtruth[0]) + simulation_params["all_seeds"][0]]
        ),
    )


def _assert_generation_preserves_rng_states(monkeypatch, *, fail=False):
    _patch_minimal_simulation(monkeypatch, fail=fail)
    original_backend_state = backend_random.get_state()
    original_numpy_state = np.random.get_state()
    try:
        backend_random.seed(101)
        np.random.seed(202)
        expected_backend_state = backend_random.get_state()
        expected_numpy_state = np.random.get_state()

        if fail:
            with pytest.raises(RuntimeError, match="generation failed"):
                simulation_module.generate_simulated_scenarios(
                    {"all_seeds": [7], "n_timesteps": 1}
                )
        else:
            simulation_module.generate_simulated_scenarios(
                {"all_seeds": [7], "n_timesteps": 1}
            )

        _assert_random_states_equal(
            backend_random.get_state(), expected_backend_state
        )
        _assert_random_states_equal(np.random.get_state(), expected_numpy_state)
    finally:
        backend_random.set_state(original_backend_state)
        np.random.set_state(original_numpy_state)


def test_generate_simulated_scenarios_preserves_rng_states(monkeypatch):
    _assert_generation_preserves_rng_states(monkeypatch)


def test_generate_simulated_scenarios_restores_rng_states_after_failure(monkeypatch):
    _assert_generation_preserves_rng_states(monkeypatch, fail=True)
