import streamlit as st

st.set_page_config(layout="wide", page_title="Methodology")

st.title("Methodology and Mathematical Foundations")
st.markdown("##### By Brian Franco")
st.markdown("This section details the quantitative models and statistical tests used throughout the application.")

st.markdown("---")
st.header("Foundational Concepts")

st.subheader("Logarithmic Returns")
st.markdown(r"""
    The first step in financial time series analysis is to convert prices ($P_t$) into returns. We use logarithmic (or continuously compounded) returns, which have convenient statistical properties.
""")
st.latex(r'''
    r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})
''')

st.subheader("Model Estimation: Maximum Likelihood")
st.markdown(r"""
    The parameters for GARCH models (e.g., $\omega, \alpha, \beta$) are found by maximizing the Log-Likelihood Function (LLF). This method finds the parameter values that make the observed data most probable. Assuming the standardized residuals $z_t$ follow a standard normal distribution, the log-likelihood for a single observation $t$ is:
""")
st.latex(r'''
    l_t = -\frac{1}{2} \left( \ln(2\pi) + \ln(\sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2} \right)
''')
st.markdown(r"The total log-likelihood for all observations is the sum $L = \sum_{t=1}^{T} l_t$. The optimizer finds the set of parameters that maximizes this function $L$.")

st.subheader("Model Selection: AIC & BIC")
st.markdown(r"""
    The Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) are used to compare the fit of different models while penalizing for complexity (number of parameters). Models with lower AIC or BIC values are generally preferred.
""")
st.latex(r'''
    AIC = -2L + 2k
''')
st.latex(r'''
    BIC = -2L + k \ln(T)
''')
st.markdown(r"Where $L$ is the maximized log-likelihood, $k$ is the number of estimated parameters, and $T$ is the number of observations.")


st.markdown("---")

st.header("GARCH-Family Models")
st.markdown(r"""
    The return of an asset $r_t$ is modeled as $r_t = \mu_t + \epsilon_t$, where $\mu_t$ is the conditional mean and $\epsilon_t = \sigma_t z_t$ is the shock term. The focus is on modeling the conditional variance, $\sigma_t^2$.
""")
st.subheader("GARCH(1,1)")
st.markdown(r"The standard GARCH(1,1) model defines the conditional variance as:")
st.latex(r'''
    \sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2
''')
st.markdown(r"""
    - $\omega$: The constant long-run variance.
    - $\epsilon_{t-1}^2$: The squared shock (news) from the previous period (the ARCH term).
    - $\sigma_{t-1}^2$: The conditional variance from the previous period (the GARCH term).
""")

st.subheader("GJR-GARCH(1,1)")
st.markdown(r"This model extends GARCH to account for the leverage effect:")
st.latex(r'''
    \sigma_t^2 = \omega + (\alpha + \gamma I_{t-1}) \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
''')
st.markdown(r"""
    - $I_{t-1}$ is an indicator function that equals 1 if $\epsilon_{t-1} < 0$ (bad news) and 0 otherwise.
    - $\gamma$: The leverage term. A positive $\gamma$ indicates that negative shocks have a larger impact on volatility.
""")

st.subheader("EGARCH(1,1)")
st.markdown(r"The Exponential GARCH model also captures leverage effects and ensures positivity by modeling the log of the variance:")
st.latex(r'''
    \ln(\sigma_t^2) = \omega + \beta \ln(\sigma_{t-1}^2) + \alpha \left| \frac{\epsilon_{t-1}}{\sigma_{t-1}} \right| + \gamma \frac{\epsilon_{t-1}}{\sigma_{t-1}}
''')
st.markdown(r"""
    - The term $\gamma \frac{\epsilon_{t-1}}{\sigma_{t-1}}$ is the asymmetric component. If $\gamma < 0$, negative shocks have a larger impact than positive shocks.
""")

st.markdown("---")

st.header("Hybrid GARCH-GRU Model")
st.markdown(r"""
    This deep learning model embeds a GARCH(1,1) process directly within a Gated Recurrent Unit (GRU) cell. This combines the structured, mean-reverting properties of GARCH with the non-linear learning capabilities of a neural network.
""")
st.subheader("Core Mechanism")
st.markdown(r"""
1. **GARCH Variance Calculation**: The conditional variance $\sigma_t^2$ is calculated using dynamically learned parameters:
    $\sigma_t^2 = \omega_{learned} + \alpha_{learned} \epsilon_{t-1}^2 + \beta_{learned} \sigma_{t-1}^2$
2. **GRU Input Formulation**: The input to the GRU at time $t$ is a vector concatenating the asset return $r_t$ and the calculated variance $\sigma_t^2$:
    $\text{Input}_t = [r_t, \sigma_t^2]$
3. **GRU State Update**: The GRU cell processes this input along with its previous hidden state ($h_{t-1}$) using its internal gate mechanism to produce a new hidden state ($h_t$).
""")
st.subheader("GRU Gate Equations")
st.markdown(r"""
- **Reset Gate ($r_t$)**: Decides how much of the past information to forget.
    $r_t = \sigma(W_r \cdot [\text{Input}_t, h_{t-1}])$
- **Update Gate ($z_t$)**: Decides what information to throw away and what new information to add.
    $z_t = \sigma(W_z \cdot [\text{Input}_t, h_{t-1}])$
- **Candidate Hidden State ($\tilde{h}_t$)**: Calculates a new candidate state based on the input and the reset gate's decision.
    $\tilde{h}_t = \tanh(W_h \cdot [\text{Input}_t, r_t \odot h_{t-1}])$
- **Final Hidden State ($h_t$)**: The final state is a linear interpolation between the previous state and the candidate state, controlled by the update gate.
    $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$
""")
st.markdown(r"""
    The final hidden state $h_t$ is then passed through a dense layer to produce the volatility forecast.
""")

st.markdown("---")
st.header("Advanced Diagnostics & Model Validation")
st.subheader("News Impact Curve (NIC)")
st.markdown(r"""
    The NIC visualizes the relationship between a past shock ($\epsilon_{t-1}$) and the resulting conditional variance ($\sigma_t^2$), holding all other variables constant. In asymmetric models like GJR-GARCH, the curve will be steeper for negative shocks.
""")
st.subheader("Value at Risk (VaR) & Backtesting")
st.markdown(r"""
    VaR is a measure of downside risk. The 1-day VaR at a confidence level of $(1-\delta)$ is calculated from the model's forecasts.
""")
st.latex(r'''
    VaR_{t+1|t} = \hat{\mu}_{t+1} + \hat{\sigma}_{t+1} F^{-1}(\delta)
''')
st.markdown(r"""
    - $\hat{\mu}_{t+1}$ and $\hat{\sigma}_{t+1}$ are the forecasted conditional mean and standard deviation for the next period.
    - $F^{-1}(\delta)$ is the quantile function (or PPF) of the assumed error distribution (e.g., Normal or Student's t) at the $\delta$ level (e.g., 0.05 for 95% VaR).
""")

st.markdown("---")
st.header("References")
st.markdown("""
- **Bollerslev, T. (1986).** *Generalized Autoregressive Conditional Heteroskedasticity.* Journal of Econometrics, 31(3), 307-327.
- **Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).** *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* [arXiv:1406.1078](https://arxiv.org/abs/1406.1078).
- **Ding, Z., Granger, C. W., & Engle, R. F. (1993).** *A long memory property of stock market returns and a new model.* Journal of empirical finance, 1(1), 83-106. (Introduces the APARCH model).
- **Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993).** *On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks.* The Journal of Finance, 48(5), 1779-1801.
- **Kupiec, P. H. (1995).** *Techniques for verifying the accuracy of risk measurement models.* The Journal of Derivatives, 3(2). (Seminal paper on VaR backtesting).
- **Mincer, J., & Zarnowitz, V. (1969).** *The evaluation of economic forecasts.* In Economic Forecasts and Expectations: Analysis of Forecasting Behavior and Performance (pp. 3-46). NBER.
- **Nelson, D. B. (1991).** *Conditional Heteroskedasticity in Asset Returns: A New Approach.* Econometrica, 59(2), 347-370.
- **Python `arch` library documentation:** [arch.readthedocs.io](https://arch.readthedocs.io/en/latest/)
- **Python `statsmodels` library documentation:** [www.statsmodels.org](https://www.statsmodels.org/stable/index.html)
""")
