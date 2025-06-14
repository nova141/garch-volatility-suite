import streamlit as st
import os
import sys

# --- PAGE CONFIGURATION ---
# This is the main entry point, so we set the overall app config here.
st.set_page_config(
    page_title="GARCH Volatility Suite",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is a GARCH volatility analysis tool."
    }
)

# --- PATH CORRECTION ---
# Ensures the 'core' module can be found by any page.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END PATH CORRECTION ---


# --- HELPER FUNCTION & SETUP ---
def load_css(file_name):
    """Loads a CSS file and injects it into the Streamlit app."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Silently ignore if CSS file is not found

load_css("assets/custom.css")


# API KEY CHECK
def check_api_key():
    """Checks for and returns the Polygon API key from Streamlit secrets."""
    if "POLYGON_API_KEY" in st.secrets:
        return st.secrets["POLYGON_API_KEY"]
    return None

# MAIN PAGE CONTENT
st.title("Garch Volatility Suite")
st.markdown("""by Brian Franco""")
st.markdown("---")
st.markdown("""
This application provides a toolkit for the analysis, forecasting, and backtesting of financial market volatility using econometric and machine learning models.

### Core Features:
- **Advanced GARCH Models**: Rigorously analyze volatility with standard GARCH, GJR-GARCH, and EGARCH models, complete with diagnostic tools.
- **Hybrid Deep Learning Model**: Leverage a cutting-edge GARCH-GRU model that combines econometric structure with neural network flexibility.
- **Robust Backtesting**: Validate model performance with out-of-sample rolling window forecasts and Value at Risk (VaR) exception analysis.
- **In-Depth Diagnostics**: Go beyond simple forecasts with tools like the News Impact Curve, QQ Plots, and Mincer-Zarnowitz regressions to ensure model integrity.

### How to Get Started:
1.  **Select an Analysis Page**: Use the navigation menu in the sidebar to choose a tool.
2.  **Configure Parameters**: On the selected page, enter an asset ticker and set the desired model parameters.
3.  **Run Analysis**: Click the "Run" button to begin the quantitative analysis.

For a detailed explanation of the models and statistical tests used, please see the **Methodology** page.
""")

# SIDEBAR CONTENT
if st.sidebar.button("Check API Key Status"):
    st.sidebar.success("Polygon.io API key is loaded.")


