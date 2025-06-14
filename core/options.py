import numpy as np
from scipy.stats import norm


# Black-Scholes-Merton Formula
def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return 0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    return price


# IV Calculation (Newton-Raphson Method)
def implied_volatility(market_price, S, K, T, r, option_type='call', tol=1e-5, max_iter=100):
    if T <= 0: return np.nan

    if option_type == 'call':
        if market_price < max(0, S - K * np.exp(-r * T)):
            return np.nan  # Price is below intrinsic value
    else:  # put
        if market_price < max(0, K * np.exp(-r * T) - S):
            return np.nan  # Price is below intrinsic value

    sigma = 0.5  # Initial guess

    for _ in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type)

        # Vega: derivative of price with respect to volatility
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)

        diff = price - market_price

        if abs(diff) < tol:
            return sigma
        if vega < 1e-6:
            return np.nan

        sigma = sigma - diff / vega

        if sigma <= 0:  # Ensure volatility remains positive
            sigma = tol

    final_price = black_scholes(S, K, T, r, sigma, option_type)
    if abs(final_price - market_price) < tol * 5:  # Looser tolerance for final check
        return sigma
    else:
        return np.nan
