"""
valuation_engine.py
Deterministic valuation algorithms, mean-reversion bands, and cross-sectional metrics.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def compute_historical_bands(pe_series: pd.Series, current_pe: float) -> Dict[str, Any]:
    """
    Computes rolling parametric and non-parametric historical valuation bounds.
    
    Formulas:
        Mean (mu) = (1/N) * sum(PE_t)
        Standard Deviation (sigma) = sqrt( (1/(N-1)) * sum((PE_t - mu)^2) )
        Upper Band = mu + 1*sigma
        Lower Band = mu - 1*sigma
        Percentile Rank = (Count(PE_t <= Current_PE) / N) * 100
    """
    clean_pe = pe_series.dropna()
    clean_pe = clean_pe[clean_pe > 0]  # Strip negative or distorted earnings multiples
    
    if clean_pe.empty or current_pe <= 0:
        return {
            "mean": np.nan, "std": np.nan, "median": np.nan,
            "upper_1s": np.nan, "lower_1s": np.nan,
            "upper_2s": np.nan, "lower_2s": np.nan,
            "min": np.nan, "max": np.nan,
            "percentile_rank": np.nan,
            "classification": "INSUFFICIENT DATA",
            "color": "#6c757d"
        }

    mu = float(clean_pe.mean())
    sigma = float(clean_pe.std(ddof=1))
    median = float(clean_pe.median())
    min_val = float(clean_pe.min())
    max_val = float(clean_pe.max())

    upper_1s = mu + sigma
    lower_1s = max(0.0, mu - sigma)
    upper_2s = mu + (2 * sigma)
    lower_2s = max(0.0, mu - (2 * sigma))

    percentile = float((clean_pe <= current_pe).mean() * 100.0)

    # Classification logic
    is_undervalued = (current_pe < lower_1s) or (current_pe < (0.85 * median))
    is_overvalued = (current_pe > upper_1s) or (current_pe > (1.15 * median))

    if is_undervalued:
        classification = "UNDERVALUED"
        color = "#00C805"  # Green
    elif is_overvalued:
        classification = "OVERVALUED"
        color = "#FF3B30"  # Red
    else:
        classification = "FAIR VALUE"
        color = "#FF9500"  # Amber

    return {
        "mean": round(mu, 2),
        "std": round(sigma, 2),
        "median": round(median, 2),
        "upper_1s": round(upper_1s, 2),
        "lower_1s": round(lower_1s, 2),
        "upper_2s": round(upper_2s, 2),
        "lower_2s": round(lower_2s, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "percentile_rank": round(percentile, 1),
        "classification": classification,
        "color": color
    }


def compute_peer_relative_spread(company_pe: float, peer_median_pe: float) -> Dict[str, Any]:
    """
    Evaluates valuation premia or discounts against an industry peer basket.
    """
    if company_pe <= 0 or peer_median_pe <= 0 or np.isnan(peer_median_pe):
        return {"relative_multiple": np.nan, "spread_pct": np.nan}

    relative_multiple = company_pe / peer_median_pe
    spread_pct = ((company_pe - peer_median_pe) / peer_median_pe) * 100.0

    return {
        "relative_multiple": round(relative_multiple, 2),
        "spread_pct": round(spread_pct, 2)
    }


def aggregate_basket_multiples(df_peers: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregates peer table multiples into basket summary metrics.
    """
    if df_peers.empty or "P/E" not in df_peers.columns:
        return {"median_pe": np.nan, "mean_pe": np.nan, "weighted_pe": np.nan}

    valid_peers = df_peers[df_peers["P/E"] > 0].copy()
    if valid_peers.empty:
        return {"median_pe": np.nan, "mean_pe": np.nan, "weighted_pe": np.nan}

    median_pe = float(valid_peers["P/E"].median())
    mean_pe = float(valid_peers["P/E"].mean())

    if "Market Cap" in valid_peers.columns and valid_peers["Market Cap"].sum() > 0:
        weighted_pe = float(
            np.average(valid_peers["P/E"], weights=valid_peers["Market Cap"])
        )
    else:
        weighted_pe = mean_pe

    return {
        "median_pe": round(median_pe, 2),
        "mean_pe": round(mean_pe, 2),
        "weighted_pe": round(weighted_pe, 2)
  }
  
