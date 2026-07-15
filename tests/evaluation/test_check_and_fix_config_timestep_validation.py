import pytest
from pyrecest.backend import eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation.check_and_fix_config import check_and_fix_config


def _base_config(n_timesteps):
    return {
        "n_timesteps": n_timesteps,
        "initial_prior": GaussianDistribution(zeros(1), eye(1)),
    }


@pytest.mark.parametrize("invalid_timesteps", [True, False, None])
def test_check_and_fix_config_rejects_noninteger_timestep_sentinels(
    invalid_timesteps,
):
    with pytest.raises(TypeError, match="n_timesteps must be an integer"):
        check_and_fix_config(_base_config(invalid_timesteps))


def test_check_and_fix_config_accepts_one_timestep():
    config = check_and_fix_config(_base_config(1))

    assert config["apply_sys_noise_times"] == [False]
    assert config["n_meas_at_individual_time_step"] == [1]
