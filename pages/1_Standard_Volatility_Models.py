import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import os
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.data_handler import get_underlying_price_history, calculate_log_returns
from core.models import fit_garch_model, calculate_news_impact_curve

st.set_page_config(layout="wide", page_title="Standard GARCH Analysis")

st.title("GARCH-Family Volatility Analysis")
st.markdown("Analyze and diagnose asset volatility using standard GARCH, GJR-GARCH, and EGARCH models.")

# Sidebar for User Inputs
with st.sidebar:
    st.header("Analysis Configuration")
    ticker = st.text_input("Enter Stock Ticker:", "SPY").upper()

    end_date_default = date.today()
    start_date_default = end_date_default - timedelta(days=5 * 365)

    date_range = st.date_input("Select Date Range:", value=(start_date_default, end_date_default))
    start_date, end_date = date_range

    model_type = st.selectbox("Select GARCH Model:", ['GARCH(1,1)', 'GJR-GARCH(1,1)', 'EGARCH(1,1)'])
    dist = st.selectbox("Error Distribution:", ['Normal', 't', 'skewt'])

    run_analysis = st.button("Run Analysis", type="primary")

# Display Results
if run_analysis:
    if not ticker:
        st.warning("Please enter a stock ticker.")
    else:
        with st.spinner(f"Fetching historical data for {ticker}..."):
            price_data = get_underlying_price_history(ticker, start_date, end_date)

        if price_data.empty:
            st.error(f"Could not retrieve data for {ticker}.")
        else:
            st.success(f"Successfully fetched {len(price_data)} data points for {ticker}.")
            log_returns = calculate_log_returns(price_data)

            with st.spinner(f"Fitting {model_type} model..."):
                garch_results = fit_garch_model(log_returns, model_name=model_type, dist=dist)

            if garch_results:
                st.success(f"{model_type} model fitted successfully.")
                cond_vol_daily = garch_results.conditional_volatility
                realized_vol_daily = log_returns.rolling(window=21).std()
                annualized_garch_vol = cond_vol_daily * np.sqrt(252)
                annualized_realized_vol = realized_vol_daily * np.sqrt(252)
                comparison_df = pd.concat([annualized_realized_vol, annualized_garch_vol], axis=1).dropna()
                comparison_df.columns = ['Realized Vol (21-Day Rolling)', 'GARCH Forecast (Annualized)']

                st.subheader("GARCH Forecast vs. Realized Volatility")
                fig_comp = go.Figure()
                # Plot Realized Volatility (rolling std dev)
                fig_comp.add_trace(
                    go.Scatter(x=comparison_df.index, y=comparison_df['Realized Vol (21-Day Rolling)'], mode='lines',
                               name='Realized Vol (21-Day Rolling)', line=dict(color='gray', dash='dash')))
                # Plot GARCH Forecast
                fig_comp.add_trace(
                    go.Scatter(x=comparison_df.index, y=comparison_df['GARCH Forecast (Annualized)'], mode='lines',
                               name=f'{model_type} Forecast (Annualized)', line=dict(color='purple')))

                fig_comp.update_layout(
                    title=f"Annualized Volatility: GARCH Forecast vs. 21-Day Realized Volatility",
                    xaxis_title="Date",
                    yaxis_title="Annualized Volatility (%)",
                    template="plotly_white",
                    legend=dict(x=0.01, y=0.99)
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                st.caption(
                    "This plot compares the model's forward-looking forecast to a backward-looking 21-day rolling standard deviation")

                with st.expander("Show Performance Metrics vs. Daily Squared Returns"):
                    daily_comparison_df = pd.concat([log_returns ** 2, cond_vol_daily], axis=1).dropna()
                    daily_comparison_df.columns = ['Daily Squared Returns', 'Daily GARCH Forecast']

                    mse = mean_squared_error(daily_comparison_df['Daily Squared Returns'],
                                             daily_comparison_df['Daily GARCH Forecast'])
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(daily_comparison_df['Daily Squared Returns'],
                                              daily_comparison_df['Daily GARCH Forecast'])

                    col_perf1, col_perf2, col_perf3 = st.columns(3)
                    col_perf1.metric("Mean Squared Error (MSE)", f"{mse:.6f}")
                    col_perf2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.6f}")
                    col_perf3.metric("Mean Absolute Error (MAE)", f"{mae:.6f}")

                st.markdown("---")
                st.subheader("Advanced Model Diagnostics")

                std_resid = garch_results.std_resid

                st.markdown("##### Standardized Residuals Over Time")
                fig_resid = go.Figure()
                fig_resid.add_trace(go.Scatter(x=std_resid.index, y=std_resid, mode='lines', name='Std. Residuals',
                                               line=dict(color='indianred', width=1)))
                fig_resid.add_hline(y=1.96, line_dash="dash", line_color="black", annotation_text="95% CI")
                fig_resid.add_hline(y=-1.96, line_dash="dash", line_color="black")
                fig_resid.update_layout(title="Standardized Residuals (Should be unpredictable noise)", height=300)
                st.plotly_chart(fig_resid, use_container_width=True)
                st.caption(
                    "Residuals should mostly stay between the black dashed lines (±1.96 standard deviations) if they are normally distributed.")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("##### News Impact Curve (NIC)")
                    nic_df = calculate_news_impact_curve(garch_results)
                    fig_nic = go.Figure()
                    fig_nic.add_trace(go.Scatter(x=nic_df['shock'], y=nic_df['conditional_variance'], mode='lines',
                                                 line=dict(color='blue')))
                    fig_nic.update_layout(xaxis_title="Shock (Past Return)", yaxis_title="Cond. Variance",
                                          title="Asymmetric Volatility Response")
                    st.plotly_chart(fig_nic, use_container_width=True)

                with col2:
                    st.markdown("##### QQ Plot of Residuals")
                    probplot = ProbPlot(std_resid, dist=garch_results.model.distribution, fit=False)
                    theoretical_quantiles = probplot.theoretical_quantiles
                    sample_quantiles = probplot.sample_quantiles
                    fig_qq = go.Figure()
                    fig_qq.add_trace(
                        go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name='Quantiles'))
                    min_val, max_val = min(min(theoretical_quantiles), min(sample_quantiles)), max(
                        max(theoretical_quantiles), max(sample_quantiles))
                    fig_qq.add_trace(
                        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Reference Line',
                                   line=dict(color='red', dash='dash')))
                    fig_qq.update_layout(title="Quantile-Quantile Plot", xaxis_title="Theoretical Quantiles",
                                         yaxis_title="Sample Quantiles")
                    st.plotly_chart(fig_qq, use_container_width=True)

                with col3:
                    st.markdown("##### ACF of Squared Residuals")
                    fig_acf, ax_acf = plt.subplots(figsize=(6, 5))
                    sm.graphics.tsa.plot_acf(std_resid ** 2, lags=40, ax=ax_acf, title="ACF of Squared Residuals")
                    ax_acf.grid(True)
                    st.pyplot(fig_acf)
                    st.caption("Checks for remaining volatility patterns.")

                with st.expander("Show Full Model Summary"):
                    st.text(str(garch_results.summary()))
            else:
                st.error("Could not generate model results.")
else:
    st.info("Configure your analysis in the sidebar and click 'Run Analysis'.")

