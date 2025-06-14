import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.data_handler import get_underlying_price_history, calculate_log_returns
from core.models import rolling_window_forecast, calculate_dynamic_var, backtest_var

st.set_page_config(layout="wide")

st.title("Rolling Forecast & VaR Backtesting")
st.markdown(
    "Rigorously backtest GARCH models using a rolling window forecast to evaluate out-of-sample performance and assess Value at Risk (VaR) model accuracy.")

# Sidebar for User Inputs
with st.sidebar:
    st.header("Backtesting Configuration")
    ticker = st.text_input("Enter Stock Ticker:", "SPY").upper()

    end_date_default = date.today()
    start_date_default = end_date_default - timedelta(days=3 * 365)

    date_range = st.date_input("Select Date Range:", value=(start_date_default, end_date_default))
    start_date, end_date = date_range

    st.header("Model Specification")
    vol_model = st.selectbox("Volatility Model:", ['GARCH(1,1)', 'EGARCH(1,1)', 'GJR-GARCH(1,1)'])
    mean_model = st.selectbox("Mean Model:", ['Constant', 'ARX'])
    dist = st.selectbox("Error Distribution:", ['Normal', 't', 'skewt'])

    st.header("Rolling Window Parameters")
    window_type = st.radio("Window Type:", ["Fixed", "Expanding"], index=0)
    expanding_window = True if window_type == "Expanding" else False

    initial_window_size = st.slider("Initial Window Size (Trading Days):", min_value=100, max_value=1000, value=252)

    run_backtest = st.button("Run Backtest", type="primary")

if run_backtest:
    if not ticker:
        st.warning("Please enter a stock ticker.")
    else:
        with st.spinner("Fetching data..."):
            price_data = get_underlying_price_history(ticker, start_date, end_date)

        if price_data.empty:
            st.error(f"Could not retrieve data for {ticker}.")
        else:
            log_returns = calculate_log_returns(price_data) * 100

            st.info(
                f"Performing rolling window forecast with an initial window of {initial_window_size} days.")

            forecast_results, final_res = rolling_window_forecast(
                log_returns, vol_model, mean_model, dist, initial_window_size, expanding=expanding_window
            )

            if forecast_results and final_res:
                st.subheader("VaR Backtest Results")

                mean_forecasts, var_forecasts = forecast_results
                consolidated_forecasts = {'mean': mean_forecasts.iloc[:, 0], 'variance': var_forecasts.iloc[:, 0]}
                var_df = calculate_dynamic_var(consolidated_forecasts, final_res)
                exceptions_95, exceptions_99 = backtest_var(log_returns, var_df)

                if exceptions_95 is not None and exceptions_99 is not None:
                    num_exceptions_95 = exceptions_95.sum()
                    num_exceptions_99 = exceptions_99.sum()
                    total_obs = len(exceptions_95)
                    breach_pct_95 = (num_exceptions_95 / total_obs) * 100
                    breach_pct_99 = (num_exceptions_99 / total_obs) * 100

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Backtest Observations", total_obs)
                    with col2:
                        st.metric("95% VaR Breaches", f"{num_exceptions_95}", f"{breach_pct_95:.2f}%")
                        st.caption("Expected: 5.0%")
                    with col3:
                        st.metric("99% VaR Breaches", f"{num_exceptions_99}", f"{breach_pct_99:.2f}%")
                        st.caption("Expected: 1.0%")

                    st.subheader("VaR Backtest Visualization")
                    plot_data = pd.concat([log_returns, var_df], axis=1).dropna()
                    breach_points_99 = plot_data[exceptions_99]

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=plot_data.index, y=plot_data.iloc[:, 0], mode='lines', name='Log Returns',
                                   line=dict(color='rgba(128, 128, 128, 0.7)', width=1.5)))
                    fig.add_trace(
                        go.Scatter(x=plot_data.index, y=plot_data['VaR_99'], mode='lines', name='99% VaR Forecast',
                                   line=dict(color='red', dash='dash')))
                    fig.add_trace(go.Scatter(x=breach_points_99.index, y=breach_points_99.iloc[:, 0], mode='markers',
                                             name='VaR Breaches', marker=dict(color='red', size=8, symbol='x')))

                    fig.update_layout(
                        title=f"99% VaR Backtest for {ticker}",
                        xaxis_title="Date",
                        yaxis_title="Returns / VaR (%)",
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to perform VaR backtest.")
            else:
                st.error("Rolling window forecast failed. The model may not have converged on the data windows.")
else:
    st.info("Configure your backtest in the sidebar and click 'Run Backtest'.")
