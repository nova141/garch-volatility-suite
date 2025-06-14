import streamlit as st
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
import numpy as np


def _fit_garch_model_uncached(log_returns: pd.Series, model_name: str, mean_model: str, dist: str):
    model_params = {
        'GARCH(1,1)': {'p': 1, 'q': 1, 'vol': 'GARCH'},
        'GJR-GARCH(1,1)': {'p': 1, 'o': 1, 'q': 1, 'vol': 'GARCH'},
        'EGARCH(1,1)': {'p': 1, 'o': 1, 'q': 1, 'vol': 'EGARCH'},
        'APARCH(1,1)': {'p': 1, 'o': 1, 'q': 1, 'vol': 'APARCH'}
    }
    params = model_params.get(model_name)
    if params is None: return None

    model = arch_model(
        log_returns,
        mean=mean_model,
        lags=1 if 'AR' in mean_model else 0,
        **params,
        dist=dist
    )
    try:
        return model.fit(update_freq=0, disp='off')
    except Exception:
        return None


@st.cache_resource(ttl="1h")
def fit_garch_model(log_returns: pd.Series, model_name: str, mean_model: str = 'Constant', dist: str = 'Normal'):
    results = _fit_garch_model_uncached(log_returns, model_name, mean_model, dist)
    if results is None:
        st.error(f"Model fitting failed for {model_name}. The model may not have converged.")
    return results


def calculate_news_impact_curve(results: ARCHModelResult):
    if results is None:
        return pd.DataFrame({'shock': [], 'conditional_variance': []})

    # Create a range of shocks (standardized residuals)
    shocks = np.linspace(-3, 3, 100)
    try:
        nic_data = results.news_impact(shocks)
        nic_df = pd.DataFrame({'shock': nic_data.index, 'conditional_variance': nic_data.iloc[:, 0].values})
    except Exception:
        omega = results.params.get('omega', 0)
        alpha = results.params.get('alpha[1]', 0)
        gamma = results.params.get('gamma[1]', 0)
        beta = results.params.get('beta[1]', 0)

        # Approximate unconditional variance to hold constant
        uncond_vol_sq = omega / (1 - alpha - 0.5 * gamma - beta)
        uncond_vol_sq = max(uncond_vol_sq, 0)  # Ensure non-negative

        # Leverage term for GJR-GARCH
        leverage = (shocks < 0) * gamma
        cond_var = omega + (alpha + leverage) * shocks ** 2 + beta * uncond_vol_sq
        nic_df = pd.DataFrame({'shock': shocks, 'conditional_variance': cond_var})

    return nic_df


def rolling_window_forecast(log_returns: pd.Series, model_name: str, mean_model: str, dist: str, window_size: int,
                            expanding: bool = False):
    forecasts = []
    T = len(log_returns)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(T - window_size):
        current_window = log_returns.iloc[:window_size + i] if expanding else log_returns.iloc[i:window_size + i]
        res = _fit_garch_model_uncached(current_window, model_name, mean_model, dist)
        if res:
            forecasts.append(res.forecast(horizon=1, reindex=False))

        progress_bar.progress((i + 1) / (T - window_size))
        status_text.text(f"Forecasting {i + 1}/{T - window_size}...")

    if not forecasts:
        status_text.error("Rolling forecast failed. The model failed to converge on every window.")
        return None, None

    status_text.success("Rolling forecast complete.")
    mean_forecasts = pd.concat([f.mean for f in forecasts])
    var_forecasts = pd.concat([f.variance for f in forecasts])

    final_res = fit_garch_model(log_returns, model_name, mean_model, dist)
    if not final_res: return None, None

    return (mean_forecasts, var_forecasts), final_res


def calculate_dynamic_var(forecasts, results: ARCHModelResult):
    if not all(k in forecasts for k in ['mean', 'variance']):
        st.error("Invalid forecast object passed to calculate_dynamic_var.")
        return pd.DataFrame()

    cond_mean = forecasts['mean']
    cond_var = forecasts['variance']

    distribution = results.model.distribution
    params = results.params

    dist_params = []
    if distribution.name == "Student's T":
        dist_params = [params['nu']]
    elif distribution.name == "Skewed Student's T":  # Corrected the name string
        dist_params = [params['eta'], params['lambda']]
    q_95 = distribution.ppf(0.05, dist_params)
    q_99 = distribution.ppf(0.01, dist_params)

    var_95 = cond_mean + np.sqrt(cond_var) * q_95
    var_99 = cond_mean + np.sqrt(cond_var) * q_99

    return pd.DataFrame({'VaR_95': var_95, 'VaR_99': var_99})


def backtest_var(log_returns, var_df):
    if var_df is None or var_df.empty:
        return None, None

    aligned_data = pd.concat([log_returns, var_df], axis=1).dropna()
    actual = aligned_data.iloc[:, 0]
    exceptions_95 = actual < aligned_data['VaR_95']
    exceptions_99 = actual < aligned_data['VaR_99']

    return exceptions_95, exceptions_99
