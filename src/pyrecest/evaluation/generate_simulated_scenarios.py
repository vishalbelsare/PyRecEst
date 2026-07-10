import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import random

from .check_and_fix_config import check_and_fix_config
from .generate_groundtruth import generate_groundtruth
from .generate_measurements import generate_measurements


def _seed_simulation_rngs(seed):
    """Seed all RNGs used by the simulation-generation helpers."""
    random.seed(seed)
    np.random.seed(seed)


def _capture_simulation_rng_states():
    """Capture caller-owned backend and NumPy RNG states."""
    return random.get_state(), np.random.get_state()


def _restore_simulation_rng_states(backend_state, numpy_state):
    """Restore caller-owned backend and NumPy RNG states."""
    random.set_state(backend_state)
    np.random.set_state(numpy_state)


def generate_simulated_scenarios(
    simulation_params,
):
    """
    Generate simulated scenarios.

    Returns
    -------
    groundtruths : numpy.ndarray
        The groundtruths.
    measurements : numpy.ndarray
        The measurements.

    """
    backend_rng_state, numpy_rng_state = _capture_simulation_rng_states()
    try:
        simulation_params = check_and_fix_config(simulation_params)
        all_seeds = simulation_params["all_seeds"]
        try:
            all_seeds = list(all_seeds)
        except TypeError:
            all_seeds = [all_seeds]
        n_runs = len(all_seeds)

        groundtruths = np.empty(
            (n_runs, simulation_params["n_timesteps"]),
            dtype=object,
        )
        measurements = np.empty(
            (n_runs, simulation_params["n_timesteps"]),
            dtype=object,
        )

        for run, seed in enumerate(all_seeds):
            _seed_simulation_rngs(seed)
            groundtruths[run, :] = generate_groundtruth(simulation_params)
            measurements[run, :] = generate_measurements(
                groundtruths[run, :], simulation_params
            )

        return groundtruths, measurements
    finally:
        _restore_simulation_rng_states(backend_rng_state, numpy_rng_state)
