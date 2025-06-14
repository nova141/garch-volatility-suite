import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
import numpy as np
import sys
import os
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.data_handler import get_underlying_price_history, calculate_log_returns, get_latest_stock_price, \
    get_option_expirations, get_options_chain, get_risk_free_rate
from core.hybrid_models import GARCH_GRU, train_hybrid_model, predict_with_hybrid_model, \
    run_mincer_zarnowitz_regression, monte_carlo_forecast_cone, calculate_saliency
from core.utils import create_sequences, scale_data
from core.options import implied_volatility

st.set_page_config(layout="wide", page_title="Hybrid GARCH-GRU Model")

st.title("Hybrid GARCH-GRU Volatility Forecasting")
st.markdown(
    "Forecast volatility using a deep learning model that combines a GARCH(1,1) process with a Gated Recurrent Unit (GRU) network.")

if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = ""

# Sidebar for User Inputs
with st.sidebar:
    st.header("Analysis Configuration")
    ticker = st.text_input("Enter Stock Ticker:", "SPY").upper()

    end_date_default = date.today()
    start_date_default = end_date_default - timedelta(days=5 * 365)
    date_range = st.date_input("Select Date Range:", value=(start_date_default, end_date_default))
    start_date, end_date = date_range

    st.header("Hybrid Model Hyperparameters")
    seq_len = st.slider("Sequence Length:", 5, 30, 15, help="Number of past trading days used to predict the next day.")
    epochs = st.number_input("Training Epochs:", 10, 500, 100)
    lr = st.number_input("Learning Rate:", 0.0001, 0.01, 0.001, format="%.4f")


    def run_button_callback():
        st.session_state.analysis_run = True
        st.session_state.last_ticker = ticker


    run_analysis_button = st.button("Train and Forecast", type="primary", on_click=run_button_callback)

# Reset state if ticker changes
if ticker != st.session_state.last_ticker:
    st.session_state.analysis_run = False

# Displaying Results
if st.session_state.analysis_run:
    if not ticker:
        st.warning("Please enter a stock ticker.")
        st.stop()


    @st.cache_data
    def get_log_returns_cached(ticker, start, end):
        price_data = get_underlying_price_history(ticker, start, end)
        if price_data.empty: return None
        return calculate_log_returns(price_data)


    @st.cache_resource
    def run_full_model_training_cached(_ticker, _log_returns, _seq_len, _epochs, _lr):
        model_instance = GARCH_GRU()
        return train_hybrid_model(model_instance, _log_returns, sequence_length=_seq_len, epochs=_epochs, lr=_lr)


    log_returns = get_log_returns_cached(ticker, start_date, end_date)

    if log_returns is None:
        st.error(f"Could not retrieve data for {ticker}.")
        st.stop()

    trained_model, vol_scaler, loss_history, param_history = run_full_model_training_cached(ticker, log_returns,
                                                                                            seq_len, epochs, lr)
    with st.spinner("Generating forecasts, forecast cone, and diagnostics..."):
        # 1-Step ahead forecast
        hybrid_predictions_daily_var = predict_with_hybrid_model(trained_model, log_returns, seq_len, vol_scaler)

        # Prepare data for multi-step forecasts
        returns_np = log_returns.values.astype(np.float32)
        X_returns, _ = create_sequences(pd.Series(returns_np), seq_len)
        X_returns_scaled, _ = scale_data(X_returns)
        last_seq_scaled = torch.from_numpy(X_returns_scaled[-1, :]).float().unsqueeze(0)
        last_sigma2 = hybrid_predictions_daily_var.iloc[-1]

        # Multi-step ahead forecast cone
        median_forecast, lower_b, upper_b = monte_carlo_forecast_cone(trained_model, last_seq_scaled, last_sigma2,
                                                                      vol_scaler)

    # MAIN VISUALIZATION
    st.subheader("GARCH-GRU Forecast vs. Realized Volatility")
    realized_vol_daily_std = log_returns.rolling(window=21).std()
    annualized_hybrid_vol = np.sqrt(np.maximum(0, hybrid_predictions_daily_var)) * np.sqrt(252)
    annualized_realized_vol = realized_vol_daily_std * np.sqrt(252)

    comparison_df = pd.concat([annualized_realized_vol, annualized_hybrid_vol], axis=1).dropna()
    comparison_df.columns = ['Realized Vol (21-Day Rolling)', 'Hybrid Forecast (Annualized)']

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Realized Vol (21-Day Rolling)'],
                                  name='Realized Vol (21-Day Rolling)', line=dict(color='gray', dash='dash')))
    fig_comp.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Hybrid Forecast (Annualized)'],
                                  name='GARCH-GRU Forecast (Annualized)', line=dict(color='purple', width=2)))
    fig_comp.update_layout(title=f"Annualized Volatility: Hybrid Forecast vs. 21-Day Realized Volatility",
                           yaxis_title="Annualized Volatility", legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig_comp, use_container_width=True)
    st.caption(
        "This plot compares the model's forward-looking forecast to a backward-looking 21-day rolling standard deviation, a common industry benchmark for realized volatility.")

    # DIAGNOSTICS
    with st.expander("Show Training Performance and Parameter Stability"):
        col1, col2 = st.columns(2)
        with col1:
            fig_loss = go.Figure(data=go.Scatter(y=loss_history, mode='lines', name='Loss'))
            fig_loss.update_layout(title="Model Training Loss", xaxis_title="Epoch", yaxis_title="MSE Loss")
            st.plotly_chart(fig_loss, use_container_width=True)
        with col2:
            fig_params = go.Figure()
            fig_params.add_trace(go.Scatter(y=param_history['omega'], name='Omega (ω)'))
            fig_params.add_trace(go.Scatter(y=param_history['alpha'], name='Alpha (α)'))
            fig_params.add_trace(go.Scatter(y=param_history['beta'], name='Beta (β)'))
            fig_params.update_layout(title="Learned GARCH Parameter Stability", xaxis_title="Epoch",
                                     yaxis_title="Parameter Value")
            st.plotly_chart(fig_params, use_container_width=True)

    with st.expander("Show Forecast Quality Diagnostics (Mincer-Zarnowitz)"):
        realized_vol_sq = log_returns ** 2
        mz_comparison_df = pd.concat([realized_vol_sq, hybrid_predictions_daily_var], axis=1).dropna()
        mz_comparison_df.columns = ['Realized Vol', 'Hybrid Forecast']
        mz_results = run_mincer_zarnowitz_regression(mz_comparison_df['Realized Vol'],
                                                     mz_comparison_df['Hybrid Forecast'])
        col_mz1, col_mz2 = st.columns(2)
        with col_mz1:
            st.metric("MZ Intercept", f"{mz_results['intercept']:.6f}",
                      help="Should be close to 0 for an unbiased forecast.")
            st.metric("MZ Slope", f"{mz_results['slope']:.4f}", help="Should be close to 1 for an efficient forecast.")
            st.metric("R-Squared", f"{mz_results['r_squared']:.4f}",
                      help="Variance in realized vol explained by forecast.")
        with col_mz2:
            fig_mz = go.Figure()
            fig_mz.add_trace(
                go.Scatter(x=mz_results['plot_data']['forecast'], y=mz_results['plot_data']['realized'], mode='markers',
                           name='Data', marker=dict(opacity=0.5)))
            fig_mz.add_trace(
                go.Scatter(x=mz_results['plot_data']['forecast'], y=mz_results['fitted_values'], mode='lines',
                           name='Regression Line', line=dict(color='red')))
            fig_mz.update_layout(title="Realized vs. Forecast (Daily Variance)", xaxis_title="Forecast Volatility",
                                 yaxis_title="Realized Volatility")
            st.plotly_chart(fig_mz, use_container_width=True)

    with st.expander("Model Interpretability: Input Saliency"):
        saliency_scores = calculate_saliency(trained_model, last_seq_scaled)
        fig_saliency = go.Figure(
            data=go.Heatmap(z=[saliency_scores], x=[f't-{i}' for i in range(seq_len, 0, -1)], y=['Saliency'],
                            colorscale='Viridis'))
        fig_saliency.update_layout(title='Input Saliency Heatmap for Final Forecast', xaxis_title='Time Lag (t-n days)')
        st.plotly_chart(fig_saliency, use_container_width=True)

    # FORECAST CONE
    st.markdown("---")
    st.subheader("Multi-Step Volatility Forecast Cone")
    annualized_median = np.sqrt(np.maximum(0, median_forecast)) * np.sqrt(252)
    annualized_lower = np.sqrt(np.maximum(0, lower_b)) * np.sqrt(252)
    annualized_upper = np.sqrt(np.maximum(0, upper_b)) * np.sqrt(252)
    forecast_horizon_days = np.arange(1, len(median_forecast) + 1)

    fig_cone = go.Figure()
    fig_cone.add_trace(
        go.Scatter(x=forecast_horizon_days, y=annualized_upper, mode='lines', line=dict(width=0), showlegend=False))
    fig_cone.add_trace(
        go.Scatter(x=forecast_horizon_days, y=annualized_lower, mode='lines', line=dict(width=0), fill='tonexty',
                   fillcolor='rgba(108, 34, 214, 0.2)', name='90% Confidence Interval'))
    fig_cone.add_trace(go.Scatter(x=forecast_horizon_days, y=annualized_median, mode='lines', name='Median Forecast',
                                  line=dict(color='purple')))
    fig_cone.update_layout(title="Forecast Cone of Uncertainty for Annualized Volatility",
                           xaxis_title="Forecast Horizon (Days)", yaxis_title="Annualized Volatility (%)")
    st.plotly_chart(fig_cone, use_container_width=True)

    # OPTIONS ANALYSIS
    st.markdown("---")
    st.subheader("Volatility Smile Analysis: GARCH Forecast vs. Market Implied Volatility")

    expirations = get_option_expirations(ticker)

    if not expirations:
        st.warning("No option expiration dates found for this ticker.")
    else:
        selected_expiry = st.selectbox("Select an Option Expiration Date:", options=expirations)

        if selected_expiry:
            with st.spinner(f"Fetching options chain and calculating implied volatility for {selected_expiry}..."):
                S = get_latest_stock_price(ticker)
                r = get_risk_free_rate()
                options_df_raw = get_options_chain(ticker, selected_expiry)

                if options_df_raw.empty:
                    st.warning(
                        f"The API did not return any options data for {selected_expiry}. This can happen for illiquid tickers or if your API key plan does not include access to historical options data.")
                else:
                    expiry_dt = datetime.strptime(selected_expiry, "%Y-%m-%d")
                    T = (expiry_dt - datetime.now()).days / 365.0
                    days_to_expiry = max(0, int(T * 365))
                    if days_to_expiry < len(median_forecast):
                        model_forecast_var = median_forecast[days_to_expiry]
                        model_forecast_vol = np.sqrt(model_forecast_var * 252)
                    else:  # Handle cases where expiry is beyond forecast cone
                        model_forecast_vol = np.sqrt(median_forecast[-1] * 252)

                    if S > 0 and T > 0:
                        options_df_raw['implied_vol'] = options_df_raw.apply(
                            lambda row: implied_volatility(row['price'], S, row['strike'], T, r, row['type']),
                            axis=1
                        )
                        options_df_final = options_df_raw.dropna(subset=['implied_vol'])

                        if options_df_final.empty:
                            st.warning(
                                "Implied volatility could not be calculated for any option in the chain. This is often due to stale market prices or arbitrage violations.")
                        else:
                            fig_smile = go.Figure()
                            calls = options_df_final[options_df_final['type'] == 'call']
                            puts = options_df_final[options_df_final['type'] == 'put']
                            fig_smile.add_trace(go.Scatter(x=calls['strike'], y=calls['implied_vol'], mode='markers',
                                                           name='Implied Vol (Calls)', marker=dict(color='blue')))
                            fig_smile.add_trace(go.Scatter(x=puts['strike'], y=puts['implied_vol'], mode='markers',
                                                           name='Implied Vol (Puts)', marker=dict(color='orange')))
                            fig_smile.add_trace(
                                go.Scatter(x=options_df_final['strike'], y=[model_forecast_vol] * len(options_df_final),
                                           mode='lines', name=f'GARCH-GRU Forecast ({model_forecast_vol:.2%})',
                                           line=dict(color='purple', dash='dash')))
                            fig_smile.update_layout(
                                title=f"Volatility Smile vs. GARCH-GRU Forecast for {ticker} (Expiry: {selected_expiry})",
                                xaxis_title="Strike Price", yaxis_title="Annualized Volatility", yaxis_tickformat='.0%',
                                legend=dict(x=0.01, y=0.99))
                            st.plotly_chart(fig_smile, use_container_width=True)

                            with st.expander("Show Options Chain Data with Implied Volatility"):
                                st.dataframe(
                                    options_df_final[['ticker', 'type', 'strike', 'price', 'implied_vol']].round(4))
else:
    st.info("Configure your analysis in the sidebar and click 'Train and Forecast' to begin.")
