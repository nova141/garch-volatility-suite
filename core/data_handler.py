import streamlit as st
import pandas as pd
import numpy as np
from polygon.rest import RESTClient
from datetime import date, timedelta

@st.cache_resource(ttl="1h")
def get_polygon_client():
    if "POLYGON_API_KEY" in st.secrets:
        api_key = st.secrets["POLYGON_API_KEY"]
        return RESTClient(api_key)
    else:
        st.error("Polygon.io API key not found.")
        st.stop()


@st.cache_data(ttl="15m")
def get_underlying_price_history(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    client = get_polygon_client()
    try:
        aggs = client.get_aggs(
            ticker=ticker.upper(), multiplier=1, timespan="day",
            from_=start_date.strftime("%Y-%m-%d"), to=end_date.strftime("%Y-%m-%d"), limit=50000
        )
        if not aggs:
            st.warning(f"No data found for ticker {ticker} in the specified date range.")
            return pd.DataFrame()
        df = pd.DataFrame(aggs)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.rename(columns={'timestamp': 'date', 'volume': 'vol'})
        df = df.set_index('date')[['open', 'high', 'low', 'close', 'vol']].sort_index()
        return df
    except Exception as e:
        st.error(f"An error occurred while fetching data for {ticker}: {e}")
        return pd.DataFrame()


def calculate_log_returns(data: pd.DataFrame) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(dtype=np.float64)
    log_returns = np.log(data['close'] / data['close'].shift(1))
    return log_returns.dropna()


@st.cache_data(ttl="5m")
def get_latest_stock_price(ticker: str) -> float:
    client = get_polygon_client()
    today_str = date.today().strftime("%Y-%m-%d")
    yesterday_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        resp = client.get_daily_open_close_agg(ticker, yesterday_str)
        return resp.close
    except Exception:
        try:
            resp = client.get_daily_open_close_agg(ticker, today_str)
            return resp.close
        except Exception as e:
            st.error(f"Could not fetch latest price for {ticker}: {e}")
            return 0.0

@st.cache_data(ttl="1h")
def get_option_expirations(ticker: str) -> list:
    client = get_polygon_client()
    try:
        contracts_generator = client.list_options_contracts(underlying_ticker=ticker.upper(), limit=1000)
        expirations = sorted(list(set(c.expiration_date for c in contracts_generator)))
        return expirations
    except Exception as e:
        st.error(f"Could not fetch option expirations for {ticker}: {e}")
        return []

@st.cache_data(ttl="5m")
def get_options_chain(ticker: str, expiration_date: str) -> pd.DataFrame:
    client = get_polygon_client()
    try:
        chain = []
        prev_trading_day = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

        contracts_generator = client.list_options_contracts(
            underlying_ticker=ticker.upper(),
            expiration_date=expiration_date,
            limit=1000
        )

        for contract in contracts_generator:
            try:
                agg = client.get_daily_open_close_agg(contract.ticker, prev_trading_day)
                price = agg.close
            except Exception:
                price = None

            chain.append({
                'ticker': contract.ticker,
                'type': contract.contract_type,
                'strike': contract.strike_price,
                'price': price,
                'expiration': contract.expiration_date
            })

        df = pd.DataFrame(chain)
        return df.dropna(subset=['price']).sort_values(by='strike').reset_index(drop=True)

    except Exception as e:
        st.error(f"Could not fetch options chain for {ticker} on {expiration_date}: {e}")
        return pd.DataFrame()


def get_risk_free_rate() -> float:
    return 0.05
