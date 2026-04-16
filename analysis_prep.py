# -*- coding: utf-8 -*-
"""
analysis_prep.py — DataFrame enrichment, data-quality assessment, and
sensitivity-analysis helpers.

Functions:
    rate_data_quality
    assess_record_quality
    map_verdict_bias
    prepare_analysis_dataframe
    collect_analysis_rows
    build_sensitivity_scenarios
    run_sensitivity_analysis
"""

import numpy as np
import pandas as pd

from constants import ANALYSIS_NUMERIC_COLUMNS, FUND_SCORE_MAX, TECH_SCORE_MAX, VAL_SCORE_MAX
from fetch import has_numeric_value
from utils_time import format_age, parse_last_updated
from settings import (
    calculate_assumption_drift,
    detect_matching_preset,
    get_model_presets,
    get_model_settings,
    normalize_model_settings,
)


# ---------------------------------------------------------------------------
# Data-quality helpers
# ---------------------------------------------------------------------------

def rate_data_quality(completeness):
    """Return 'High', 'Medium', or 'Low' based on *completeness* fraction."""
    if completeness >= 0.85:
        return "High"
    if completeness >= 0.60:
        return "Medium"
    return "Low"


def assess_record_quality(record):
    """
    Return (completeness_fraction, missing_count, quality_label) for the
    key metric fields in *record*.
    """
    quality_fields = [
        "Price",
        "Sector",
        "PE_Ratio",
        "Forward_PE",
        "Profit_Margins",
        "ROE",
        "Debt_to_Equity",
        "Revenue_Growth",
        "Current_Ratio",
        "RSI",
        "MACD_Value",
        "Momentum_1M",
        "Target_Mean_Price",
        "Recommendation_Key",
    ]
    missing_count = 0
    for field in quality_fields:
        value = record.get(field)
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "N/A":
            missing_count += 1
    completeness = 1 - (missing_count / len(quality_fields))
    return completeness, missing_count, rate_data_quality(completeness)


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------

def map_verdict_bias(verdict):
    """Return 'Bullish', 'Bearish', or 'Neutral' for a verdict string."""
    if verdict in {"BUY", "STRONG BUY"}:
        return "Bullish"
    if verdict in {"SELL", "STRONG SELL"}:
        return "Bearish"
    return "Neutral"


# ---------------------------------------------------------------------------
# DataFrame enrichment
# ---------------------------------------------------------------------------

def prepare_analysis_dataframe(df, settings=None):
    """
    Enrich a raw database DataFrame with computed columns (Composite Score,
    Target Upside, Graham Discount, DCF Upside, data-quality stats,
    freshness, etc.) and sort by recency.

    Returns an empty DataFrame when *df* is None or empty.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    active_settings = get_model_settings() if settings is None else settings
    enriched = df.copy()
    for column in ANALYSIS_NUMERIC_COLUMNS:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    score_columns = [
        column for column in ["Score_Tech", "Score_Fund", "Score_Val", "Score_Sentiment"]
        if column in enriched.columns
    ]
    if "Composite Score" not in enriched.columns:
        enriched["Composite Score"] = np.nan
    if score_columns:
        score_weights = {
            "Score_Tech": active_settings["weight_technical"] * (FUND_SCORE_MAX / TECH_SCORE_MAX),
            "Score_Fund": active_settings["weight_fundamental"],
            "Score_Val": active_settings["weight_valuation"] * (FUND_SCORE_MAX / VAL_SCORE_MAX),
            "Score_Sentiment": active_settings["weight_sentiment"],
        }
        composite_score = pd.Series(0.0, index=enriched.index)
        for column in score_columns:
            composite_score = composite_score + enriched[column].fillna(0) * score_weights.get(column, 1.0)
        enriched["Composite Score"] = composite_score

    if "Target Upside" not in enriched.columns:
        enriched["Target Upside"] = np.nan
    if {"Price", "Target_Mean_Price"}.issubset(enriched.columns):
        enriched["Target Upside"] = np.where(
            enriched["Price"].notna() & (enriched["Price"] != 0),
            (enriched["Target_Mean_Price"] - enriched["Price"]) / enriched["Price"],
            np.nan,
        )

    if "Graham Discount" not in enriched.columns:
        enriched["Graham Discount"] = np.nan
    if {"Price", "Graham_Number"}.issubset(enriched.columns):
        enriched["Graham Discount"] = np.where(
            enriched["Price"].notna() & (enriched["Price"] != 0) & enriched["Graham_Number"].notna(),
            (enriched["Graham_Number"] - enriched["Price"]) / enriched["Price"],
            np.nan,
        )

    if "DCF Upside" not in enriched.columns:
        enriched["DCF Upside"] = np.nan
    if {"Price", "DCF_Intrinsic_Value"}.issubset(enriched.columns):
        enriched["DCF Upside"] = np.where(
            enriched["Price"].notna() & (enriched["Price"] != 0) & enriched["DCF_Intrinsic_Value"].notna(),
            (enriched["DCF_Intrinsic_Value"] - enriched["Price"]) / enriched["Price"],
            np.nan,
        )

    # Fill / add optional columns with safe defaults
    str_fill = {
        "Assumption_Profile": "Legacy",
        "Assumption_Fingerprint": "Legacy",
        "Market_Regime": "Unknown",
        "Decision_Notes": "",
        "Industry": "Unknown",
        "Stock_Type": "Legacy",
        "Cap_Bucket": "Unknown",
        "Style_Tags": "",
        "Type_Strategy": "",
        "Engine_Weight_Profile": "",
        "Peer_Group_Label": "",
        "Peer_Tickers": "",
        "Peer_Summary": "",
        "Peer_Comparison": "",
        "Risk_Flags": "",
        "DCF_Source": "Unavailable",
        "DCF_Confidence": "low",
        "DCF_Last_Updated": "",
        "DCF_Assumptions": "",
        "DCF_History": "",
        "DCF_Projection": "",
        "DCF_Sensitivity": "",
        "DCF_Guidance_Excerpts": "",
        "DCF_Guidance_Summary": "",
        "Sentiment_Summary": "",
        "Event_Study_Summary": "",
        "Event_Study_Events": "",
        "Last_Data_Update": "",
    }
    nan_fill = {
        "Decision_Confidence": np.nan,
        "Type_Confidence": np.nan,
        "Peer_Count": np.nan,
        "Relative_Strength_3M": np.nan,
        "Relative_Strength_6M": np.nan,
        "Relative_Strength_1Y": np.nan,
        "Event_Study_Count": np.nan,
        "Event_Study_Avg_Abnormal_1D": np.nan,
        "Event_Study_Avg_Abnormal_5D": np.nan,
    }
    for col, default in str_fill.items():
        if col not in enriched.columns:
            enriched[col] = default
        else:
            enriched[col] = enriched[col].fillna(default)
    for col, default in nan_fill.items():
        if col not in enriched.columns:
            enriched[col] = default

    if (
        "Data_Completeness" not in enriched.columns
        or "Missing_Metric_Count" not in enriched.columns
        or "Data_Quality" not in enriched.columns
        or enriched["Data_Completeness"].isna().any()
        or enriched["Missing_Metric_Count"].isna().any()
        or enriched["Data_Quality"].isna().any()
    ):
        quality_stats = enriched.apply(
            lambda row: assess_record_quality(row.to_dict()),
            axis=1,
            result_type="expand",
        )
        quality_stats.columns = ["_Data_Completeness", "_Missing_Metric_Count", "_Data_Quality"]
        enriched["Data_Completeness"] = pd.to_numeric(
            enriched.get("Data_Completeness", quality_stats["_Data_Completeness"]),
            errors="coerce",
        ).fillna(quality_stats["_Data_Completeness"])
        enriched["Missing_Metric_Count"] = pd.to_numeric(
            enriched.get("Missing_Metric_Count", quality_stats["_Missing_Metric_Count"]),
            errors="coerce",
        ).fillna(quality_stats["_Missing_Metric_Count"])
        enriched["Data_Quality"] = enriched.get("Data_Quality", quality_stats["_Data_Quality"]).fillna(
            quality_stats["_Data_Quality"]
        )

    if "Last_Updated" in enriched.columns:
        enriched["Last_Updated_Parsed"] = enriched["Last_Updated"].map(parse_last_updated)
        enriched["Freshness"] = enriched["Last_Updated"].map(format_age)
        sort_columns = ["Last_Updated_Parsed"]
        ascending = [False]
        if "Composite Score" in enriched.columns:
            sort_columns.append("Composite Score")
            ascending.append(False)
        if "Ticker" in enriched.columns:
            sort_columns.append("Ticker")
            ascending.append(True)
        enriched = enriched.sort_values(sort_columns, ascending=ascending, na_position="last")
    elif "Composite Score" in enriched.columns:
        enriched = enriched.sort_values(["Composite Score", "Ticker"], ascending=[False, True], na_position="last")
    if "Freshness" not in enriched.columns:
        enriched["Freshness"] = "Unknown"

    return enriched.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Batch collection
# ---------------------------------------------------------------------------

def collect_analysis_rows(analyst, db, tickers, refresh_live=False):
    """
    Collect analysis records for *tickers*, running live analysis when no
    saved record exists or *refresh_live* is True.

    Returns (enriched_df, failed_list, failure_reasons_dict, refreshed_list,
    cached_list).
    """
    rows = []
    failed = []
    failure_reasons = {}
    refreshed = []
    cached = []

    for ticker in tickers:
        existing = db.get_analysis(ticker)
        if refresh_live or existing.empty:
            record = analyst.analyze(ticker)
            if record is None:
                failed.append(ticker)
                failure_reasons[ticker] = analyst.last_error or "No usable market data was returned for this ticker."
                continue
            rows.append(record)
            refreshed.append(ticker)
        else:
            rows.append(existing.iloc[0].to_dict())
            cached.append(ticker)

    return prepare_analysis_dataframe(pd.DataFrame(rows)), failed, failure_reasons, refreshed, cached


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def build_sensitivity_scenarios(base_settings):
    """
    Return a list of (scenario_name, settings_dict) pairs covering the
    standard preset variants plus a valuation-strict and low-sentiment-weight
    variant derived from *base_settings*.
    """
    conservative = get_model_presets()["Conservative"]
    balanced = get_model_presets()["Balanced"]
    aggressive = get_model_presets()["Aggressive"]

    valuation_strict, _ = normalize_model_settings(
        {
            **base_settings,
            "valuation_benchmark_scale": base_settings["valuation_benchmark_scale"] * 0.95,
            "valuation_under_score_threshold": base_settings["valuation_under_score_threshold"] + 1,
            "overall_buy_threshold": base_settings["overall_buy_threshold"] + 1,
            "overall_strong_buy_threshold": base_settings["overall_strong_buy_threshold"] + 1,
        }
    )
    sentiment_light, _ = normalize_model_settings(
        {
            **base_settings,
            "weight_sentiment": max(0.5, base_settings["weight_sentiment"] - 0.3),
            "sentiment_analyst_boost": max(0.0, base_settings["sentiment_analyst_boost"] - 0.5),
        }
    )

    return [
        ("Current", base_settings),
        ("Balanced", balanced),
        ("Conservative", conservative),
        ("Aggressive", aggressive),
        ("Valuation Strict", valuation_strict),
        ("Low Sentiment Weight", sentiment_light),
    ]


def run_sensitivity_analysis(analyst, ticker, settings=None):
    """
    Run the stock-type framework across multiple assumption scenarios and
    return (scenarios_df, summary_dict) — or (None, None) on failure.
    """
    active_settings = get_model_settings() if settings is None else settings
    hist, info, news = analyst.get_data(ticker)
    if hist is None or hist.empty:
        return None, None

    scenario_rows = []
    for scenario_name, scenario_settings in build_sensitivity_scenarios(active_settings):
        record = analyst.analyze(
            ticker,
            settings=scenario_settings,
            persist=False,
            preloaded=(hist, info, news),
        )
        if record is None:
            continue
        scenario_rows.append(
            {
                "Scenario": scenario_name,
                "Preset": detect_matching_preset(scenario_settings),
                "Verdict": record["Verdict_Overall"],
                "Bias": map_verdict_bias(record["Verdict_Overall"]),
                "Overall Score": record["Overall_Score"],
                "Technical": record["Score_Tech"],
                "Fundamental": record["Score_Fund"],
                "Valuation": record["Score_Val"],
                "Sentiment": record["Score_Sentiment"],
                "Consistency": record.get("Decision_Confidence"),
                "Regime": record.get("Market_Regime"),
                "Assumption Drift": calculate_assumption_drift(scenario_settings),
                "Fingerprint": record["Assumption_Fingerprint"],
            }
        )

    if not scenario_rows:
        return None, None

    scenarios_df = pd.DataFrame(scenario_rows)
    bias_counts = scenarios_df["Bias"].value_counts()
    dominant_bias = bias_counts.index[0]
    robustness_ratio = bias_counts.iloc[0] / len(scenarios_df)
    if robustness_ratio >= 0.85:
        robustness_label = "High"
    elif robustness_ratio >= 0.60:
        robustness_label = "Medium"
    else:
        robustness_label = "Low"

    summary = {
        "dominant_bias": dominant_bias,
        "robustness_ratio": robustness_ratio,
        "robustness_label": robustness_label,
        "verdict_count": scenarios_df["Verdict"].nunique(),
        "scenario_count": len(scenarios_df),
    }
    return scenarios_df, summary
