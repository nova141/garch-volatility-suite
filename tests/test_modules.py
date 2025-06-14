import pytest
import pandas as pd
import numpy as np
from arch.univariate.base import ARCHModelResult
from core.models import fit_garch_model


@pytest.fixture
def sample_log_returns():
    np.random.seed(42)
    n_obs = 250
    returns = np.random.randn(n_obs) * 0.5
    returns[50:100] *= 3
    returns[200:220] *= 2.5
    return pd.Series(returns)

def test_fit_garch_model_returns_valid_result(sample_log_returns):
    log_returns = sample_log_returns
    result = fit_garch_model(log_returns, model_name='GARCH(1,1)')
    assert result is not None, "Model fitting should not return None for valid inputs."
    assert isinstance(result, ARCHModelResult), "The function should return an ARCHModelResult object."

def test_fit_garch_model_handles_unknown_model(sample_log_returns):
    log_returns = sample_log_returns
    result = fit_garch_model(log_returns, model_name='UNKNOWN_MODEL(1,1)')
    assert result is None, "Function should return None for an unknown model type."
