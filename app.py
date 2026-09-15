"""
app.py
Main Streamlit execution module.
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from data_fetcher import (
    fetch_index_data,
    fetch_company_data,
    scrape_screener_peers,
    INDEX_TICKER_MAP
)
from valuation_engine import (
    compute_historical_bands,
    compute_peer_relative_spread,
    aggregate_basket_multiples
)

st.set_page_config(
    page_title="Multi-Tier Equity Valuation Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Institutional Theme Styling
st.markdown("""
<style>
    .metric-card {
        background: #11141a;
        border: 1px solid #232730;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #f0f6fc;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def plot_valuation_bands(df: pd.DataFrame, bands: dict, title_label: str) -> go.Figure:
    """Renders dual-axis interactive time series with standard deviation and mean bands."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Price Curve (Secondary Axis)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Price"],
            name="Asset Price",
            line=dict(color="#4A90E2", width=1.5),
            opacity=0.65
        ),
        secondary_y=True
    )

    # 2. Historical P/E Line (Primary Axis)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["P/E"],
            name="Historical P/E",
            line=dict(color="#FFFFFF", width=2.0)
        ),
        secondary_y=False
    )

    # 3. Horizontal Statistical Thresholds
    mu = bands["mean"]
    u1, l1 = bands["upper_1s"], bands["lower_1s"]
    u2, l2 = bands["upper_2s"], bands["lower_2s"]
    med = bands["median"]

    fig.add_hline(y=mu, line_dash="dash", line_color="#8E8E93", annotation_text="10Y Mean (μ)", secondary_y=False)
    fig.add_hline(y=u1, line_dash="dot", line_color="#FF3B30", annotation_text="+1σ", secondary_y=False)
    fig.add_hline(y=l1, line_dash="dot", line_color="#34C759", annotation_text="-1σ", secondary_y=False)
    fig.add_hline(y=med, line_dash="dashdot", line_color="#FF9500", annotation_text="Median", secondary_y=False)

    fig.update_layout(
        title=f"<b>{title_label} — 10-Year Historical P/E & Price Trajectory</b>",
        height=540,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="#0b0e14",
        paper_bgcolor="#0b0e14",
        font=dict(color="#c9d1d9", family="sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1e222b")
    fig.update_yaxes(title_text="<b>P/E Multiple</b>", showgrid=True, gridwidth=1, gridcolor="#1e222b", secondary_y=False)
    fig.update_yaxes(title_text="<b>Price (₹)</b>", showgrid=False, secondary_y=True)

    return fig


# Sidebar Navigation & Selection
st.sidebar.markdown("### 🏛️ Tier Selector")
analysis_tier = st.sidebar.selectbox(
    "Select Aggregation Level",
    ["Broad Market", "Sector Specific", "Company Specific"]
)

# Data Ingestion Routing
if analysis_tier == "Broad Market":
    market_choices = ["Nifty 50", "Nifty Next 50", "Nifty 500"]
    selected_index = st.sidebar.selectbox("Benchmark Universe", market_choices)
    hist_df, curr_pe, meta = fetch_index_data(selected_index)
    title_display = selected_index
    peers_df, peer_median = pd.DataFrame(), np.nan

elif analysis_tier == "Sector Specific":
    sector_choices = [k for k in INDEX_TICKER_MAP.keys() if k not in ["Nifty 50", "Nifty Next 50", "Nifty 500"]]
    selected_sector = st.sidebar.selectbox("Industry Sub-Index", sector_choices)
    hist_df, curr_pe, meta = fetch_index_data(selected_sector)
    title_display = selected_sector
    peers_df, peer_median = pd.DataFrame(), np.nan

else:
    DEFAULT_TICKERS = ["TCS", "RELIANCE", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "LT", "TRENT", "DIXON"]
    symbol_input = st.sidebar.text_input("Enter NSE Ticker Symbol", value="TCS").upper().strip()
    hist_df, curr_pe, meta = fetch_company_data(symbol_input)
    peers_df, peer_median = scrape_screener_peers(symbol_input)
    title_display = f"{meta['name']} ({meta['symbol']})"

# Mathematical Calculations
bands = compute_historical_bands(hist_df["P/E"], curr_pe)
spread = compute_peer_relative_spread(curr_pe, peer_median)

# Control Bar Header
st.title(f"{title_display}")
st.caption(f"Valuation Engine: 10-Year Rolling Mean Reversion Bands & Multi-Timeframe Multiples")

# KPI Summary Row
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Current P/E</div>
        <div class="metric-value">{curr_pe:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">10Y Median P/E</div>
        <div class="metric-value">{bands['median']:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">10Y Mean (μ)</div>
        <div class="metric-value">{bands['mean']:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Normal Band (±1σ)</div>
        <div class="metric-value" style="font-size: 1.25rem; padding-top: 5px;">
            {bands['lower_1s']:.1f} – {bands['upper_1s']:.1f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">10Y Percentile</div>
        <div class="metric-value">{bands['percentile_rank']:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

with k6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Valuation Zone</div>
        <div style="margin-top: 6px;">
            <span class="status-badge" style="background-color: {bands['color']}22; color: {bands['color']}; border: 1px solid {bands['color']};">
                {bands['classification']}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Primary Visual Panel
fig_chart = plot_valuation_bands(hist_df, bands, title_display)
st.plotly_chart(fig_chart, use_container_width=True)

# Cross-Sectional Peer Basket Section
if analysis_tier == "Company Specific" and not peers_df.empty:
    st.subheader("Industry Peer Comparison (NSE/BSE)")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.dataframe(
            peers_df.style.format({
                "Market Cap": "₹{:,.0f} Cr",
                "P/E": "{:.2f}",
                "P/B": "{:.2f}",
                "ROCE_3Y": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Industry Median P/E</div>
            <div class="metric-value">{peer_median if not pd.isna(peer_median) else 'N/A'}</div>
            <div style="margin-top: 10px; font-size: 0.85rem; color: #8b949e;">
                Multiple vs. Peer Median: <b>{spread['relative_multiple']}x</b><br>
                Premium / Discount: <b>{spread['spread_pct']:+.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
  
