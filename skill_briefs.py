# -*- coding: utf-8 -*-
"""
skill_briefs.py — Markdown brief builders for Claude Code skills.

Each function returns a UTF-8 markdown string pre-formatted as input for the
corresponding Claude Code skill from the financial-services-plugins repo.

No streamlit imports. Depends on utils_fmt.
"""

import math

from utils_fmt import normalize_ticker, safe_json_loads


# ---------------------------------------------------------------------------
# Private formatting helper
# ---------------------------------------------------------------------------

def _fmt(value, fmt="{}", fallback="N/A"):
    """Format *value* using *fmt* or return *fallback* on None / NaN."""
    if value is None:
        return fallback
    try:
        v = float(value)
        if math.isnan(v):
            return fallback
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(value) if value else fallback


# ---------------------------------------------------------------------------
# Skill brief builders
# ---------------------------------------------------------------------------

def build_earnings_skill_brief(record):
    """Build a Markdown input brief for the ``/equity-research:earnings`` skill.

    Parameters
    ----------
    record:
        A dict or pandas Series representing a saved analysis row.

    Returns
    -------
    str
        Markdown text, or an empty string when *record* is None.
    """
    if record is None:
        return ""
    r = record if isinstance(record, dict) else record.to_dict()
    ticker = normalize_ticker(r.get("Ticker"))
    peer_comparison = safe_json_loads(r.get("Peer_Comparison"), default={})
    peers_str = str(r.get("Peer_Tickers") or "N/A")

    lines = [
        f"# Earnings Analysis Brief — {ticker}",
        "_Generated from brazingtoncompiler analysis snapshot. Use as input for `/equity-research:earnings`._",
        "",
        "## Company Overview",
        f"- **Ticker:** {ticker}",
        f"- **Sector:** {r.get('Sector') or 'N/A'}",
        f"- **Industry:** {r.get('Industry') or 'N/A'}",
        f"- **Stock Type:** {r.get('Stock_Type') or 'N/A'}  |  **Cap Bucket:** {r.get('Cap_Bucket') or 'N/A'}",
        f"- **Market Cap:** {_fmt(r.get('Market_Cap'), '${:,.0f}M')}",
        f"- **Current Price:** {_fmt(r.get('Price'), '${:.2f}')}",
        f"- **52-Week Range Position:** {_fmt(r.get('Range_Position_52W'), '{:.1%}')}",
        "",
        "## Model Verdict & Scores",
        f"- **Overall Verdict:** {r.get('Verdict_Overall') or 'N/A'}",
        f"- **Fundamental:** {r.get('Verdict_Fundamental') or 'N/A'}  (Score: {r.get('Score_Fund') or 'N/A'})",
        f"- **Valuation:** {r.get('Verdict_Valuation') or 'N/A'}  (Score: {r.get('Score_Val') or 'N/A'})",
        f"- **Technical:** {r.get('Verdict_Technical') or 'N/A'}  (Score: {r.get('Score_Tech') or 'N/A'})",
        f"- **Sentiment:** {r.get('Verdict_Sentiment') or 'N/A'}  (Score: {r.get('Score_Sentiment') or 'N/A'})",
        f"- **Market Regime:** {r.get('Market_Regime') or 'N/A'}",
        f"- **Decision Notes:** {r.get('Decision_Notes') or 'N/A'}",
        "",
        "## Valuation Multiples",
        f"- **Trailing P/E:** {_fmt(r.get('PE_Ratio'), '{:.1f}x')}",
        f"- **Forward P/E:** {_fmt(r.get('Forward_PE'), '{:.1f}x')}",
        f"- **PEG Ratio:** {_fmt(r.get('PEG_Ratio'), '{:.2f}')}",
        f"- **P/S:** {_fmt(r.get('PS_Ratio'), '{:.1f}x')}  |  **P/B:** {_fmt(r.get('PB_Ratio'), '{:.1f}x')}",
        f"- **EV/EBITDA:** {_fmt(r.get('EV_EBITDA'), '{:.1f}x')}",
        f"- **Graham Number:** {_fmt(r.get('Graham_Number'), '${:.2f}')}",
        f"- **DCF Intrinsic Value:** {_fmt(r.get('DCF_Intrinsic_Value'), '${:.2f}')}  (Upside: {_fmt(r.get('DCF_Upside'), '{:.1%}')})",
        "",
        "## Fundamentals",
        f"- **Revenue Growth (YoY):** {_fmt(r.get('Revenue_Growth'), '{:.1%}')}",
        f"- **Profit Margin:** {_fmt(r.get('Profit_Margins'), '{:.1%}')}",
        f"- **ROE:** {_fmt(r.get('ROE'), '{:.1%}')}",
        f"- **Debt/Equity:** {_fmt(r.get('Debt_to_Equity'), '{:.2f}')}",
        f"- **Current Ratio:** {_fmt(r.get('Current_Ratio'), '{:.2f}')}",
        f"- **Quality Score:** {_fmt(r.get('Quality_Score'), '{:.1f}')}",
        f"- **Dividend Yield:** {_fmt(r.get('Dividend_Yield'), '{:.2%}')}",
        "",
        "## Technical Indicators",
        f"- **RSI (14d):** {_fmt(r.get('RSI'), '{:.1f}')}",
        f"- **MACD Signal:** {r.get('MACD_Signal') or 'N/A'}",
        f"- **Trend Strength:** {_fmt(r.get('Trend_Strength'), '{:.2f}')}",
        f"- **Momentum 1M (risk-adj):** {_fmt(r.get('Momentum_1M_Risk_Adjusted'), '{:.2f}')}",
        f"- **Beta:** {_fmt(r.get('Equity_Beta'), '{:.2f}')}",
        f"- **Volatility 1M / 1Y:** {_fmt(r.get('Volatility_1M'), '{:.1%}')} / {_fmt(r.get('Volatility_1Y'), '{:.1%}')}",
        "",
        "## Analyst Coverage",
        f"- **Consensus Recommendation:** {r.get('Recommendation_Key') or 'N/A'}",
        f"- **Mean Target Price:** {_fmt(r.get('Target_Mean_Price'), '${:.2f}')}",
        f"- **Analyst Count:** {r.get('Analyst_Opinions') or 'N/A'}",
        "",
        "## Peer Group",
        f"- **Peer Group:** {r.get('Peer_Group_Label') or 'N/A'}",
        f"- **Peer Tickers:** {peers_str}",
    ]
    if peer_comparison:
        lines.append("- **Peer Comparison (vs. group averages):**")
        for metric, vals in peer_comparison.items():
            if isinstance(vals, dict):
                subj = vals.get("subject", "N/A")
                avg = vals.get("average", "N/A")
                lines.append(f"  - {metric}: Subject={subj}, Peer Avg={avg}")
    lines += [
        "",
        "## Risk Flags",
        f"{r.get('Risk_Flags') or 'None identified.'}",
        "",
        "## Sentiment",
        f"- **Headline Count:** {r.get('Sentiment_Headline_Count') or 'N/A'}",
        f"- **Sentiment Summary:** {r.get('Sentiment_Summary') or 'N/A'}",
        "",
        "---",
        "<!-- Skill: /equity-research:earnings -->",
        "<!-- Usage: In Claude Code, run /equity-research:earnings and attach this file as context when prompted for company data. -->",
    ]
    return "\n".join(lines)


def build_comps_skill_brief(record):
    """Build a Markdown input brief for the ``/financial-analysis:comps-analysis`` skill.

    Parameters
    ----------
    record:
        A dict or pandas Series representing a saved analysis row.

    Returns
    -------
    str
        Markdown text, or an empty string when *record* is None.
    """
    if record is None:
        return ""
    r = record if isinstance(record, dict) else record.to_dict()
    ticker = normalize_ticker(r.get("Ticker"))
    peer_comparison = safe_json_loads(r.get("Peer_Comparison"), default={})
    peer_tickers = [t.strip() for t in str(r.get("Peer_Tickers") or "").split(",") if t.strip()]

    lines = [
        f"# Comparable Company Analysis Brief — {ticker}",
        "_Use as input for `/financial-analysis:comps-analysis`._",
        "",
        "## Subject Company",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Ticker | {ticker} |",
        f"| Sector | {r.get('Sector') or 'N/A'} |",
        f"| Industry | {r.get('Industry') or 'N/A'} |",
        f"| Market Cap | {_fmt(r.get('Market_Cap'), '${:,.0f}M')} |",
        f"| Trailing P/E | {_fmt(r.get('PE_Ratio'), '{:.1f}x')} |",
        f"| Forward P/E | {_fmt(r.get('Forward_PE'), '{:.1f}x')} |",
        f"| EV/EBITDA | {_fmt(r.get('EV_EBITDA'), '{:.1f}x')} |",
        f"| P/S | {_fmt(r.get('PS_Ratio'), '{:.1f}x')} |",
        f"| P/B | {_fmt(r.get('PB_Ratio'), '{:.1f}x')} |",
        f"| Revenue Growth | {_fmt(r.get('Revenue_Growth'), '{:.1%}')} |",
        f"| Profit Margin | {_fmt(r.get('Profit_Margins'), '{:.1%}')} |",
        f"| ROE | {_fmt(r.get('ROE'), '{:.1%}')} |",
        f"| Debt/Equity | {_fmt(r.get('Debt_to_Equity'), '{:.2f}')} |",
        "",
        f"## Peer Group: {r.get('Peer_Group_Label') or 'N/A'}",
        f"Peers: {', '.join(peer_tickers) if peer_tickers else 'N/A'}",
    ]
    if peer_comparison:
        lines += [
            "",
            "## Peer Comparison vs. Group Averages",
            "| Metric | Subject | Peer Avg |",
            "|--------|---------|----------|",
        ]
        for metric, vals in peer_comparison.items():
            if isinstance(vals, dict):
                subj = vals.get("subject", "N/A")
                avg = vals.get("average", "N/A")
                lines.append(f"| {metric} | {subj} | {avg} |")
    lines += [
        "",
        "---",
        "<!-- Skill: /financial-analysis:comps-analysis -->",
        "<!-- Usage: In Claude Code, run /financial-analysis:comps-analysis and attach this file as context. -->",
    ]
    return "\n".join(lines)


def build_dcf_skill_brief(record):
    """Build a Markdown input brief for the ``/financial-analysis:dcf-model`` skill.

    Parameters
    ----------
    record:
        A dict or pandas Series representing a saved analysis row.

    Returns
    -------
    str
        Markdown text, or an empty string when *record* is None.
    """
    if record is None:
        return ""
    r = record if isinstance(record, dict) else record.to_dict()
    ticker = normalize_ticker(r.get("Ticker"))
    dcf_history = safe_json_loads(r.get("DCF_History"), default=[])
    dcf_projection = safe_json_loads(r.get("DCF_Projection"), default=[])
    dcf_sensitivity = safe_json_loads(r.get("DCF_Sensitivity"), default=[])

    lines = [
        f"# DCF Model Input Brief — {ticker}",
        "_Use as input for `/financial-analysis:dcf-model`._",
        "",
        "## Company",
        f"- **Ticker:** {ticker}  |  **Sector:** {r.get('Sector') or 'N/A'}  |  **Industry:** {r.get('Industry') or 'N/A'}",
        f"- **Current Price:** {_fmt(r.get('Price'), '${:.2f}')}  |  **Market Cap:** {_fmt(r.get('Market_Cap'), '${:,.0f}M')}",
        "",
        "## DCF Assumptions",
        "| Assumption | Value |",
        "|------------|-------|",
        f"| WACC | {_fmt(r.get('DCF_WACC'), '{:.2%}')} |",
        f"| Risk-Free Rate | {_fmt(r.get('DCF_Risk_Free_Rate'), '{:.2%}')} |",
        f"| Beta | {_fmt(r.get('DCF_Beta'), '{:.2f}')} |",
        f"| Cost of Equity | {_fmt(r.get('DCF_Cost_of_Equity'), '{:.2%}')} |",
        f"| Cost of Debt | {_fmt(r.get('DCF_Cost_of_Debt'), '{:.2%}')} |",
        f"| Equity Weight | {_fmt(r.get('DCF_Equity_Weight'), '{:.1%}')} |",
        f"| Debt Weight | {_fmt(r.get('DCF_Debt_Weight'), '{:.1%}')} |",
        f"| Growth Rate (explicit period) | {_fmt(r.get('DCF_Growth_Rate'), '{:.2%}')} |",
        f"| Terminal Growth Rate | {_fmt(r.get('DCF_Terminal_Growth'), '{:.2%}')} |",
        f"| Base FCF | {_fmt(r.get('DCF_Base_FCF'), '${:,.1f}M')} |",
        f"| Historical FCF Growth | {_fmt(r.get('DCF_Historical_FCF_Growth'), '{:.2%}')} |",
        f"| Historical Revenue Growth | {_fmt(r.get('DCF_Historical_Revenue_Growth'), '{:.2%}')} |",
        f"| Guidance Growth | {_fmt(r.get('DCF_Guidance_Growth'), '{:.2%}')} |",
        f"| Data Source | {r.get('DCF_Source') or 'N/A'} |",
        f"| Confidence | {r.get('DCF_Confidence') or 'N/A'} |",
        "",
        "## DCF Results",
        f"- **Intrinsic Value per Share:** {_fmt(r.get('DCF_Intrinsic_Value'), '${:.2f}')}",
        f"- **Implied Upside:** {_fmt(r.get('DCF_Upside'), '{:.1%}')}",
        f"- **Enterprise Value:** {_fmt(r.get('DCF_Enterprise_Value'), '${:,.1f}M')}",
        f"- **Equity Value:** {_fmt(r.get('DCF_Equity_Value'), '${:,.1f}M')}",
    ]
    if dcf_history:
        lines += ["", "## Historical FCF / Revenue", "| Year | FCF | Revenue |", "|------|-----|---------|"]
        for entry in dcf_history:
            if isinstance(entry, dict):
                lines.append(f"| {entry.get('year', '?')} | {entry.get('fcf', 'N/A')} | {entry.get('revenue', 'N/A')} |")
    if dcf_projection:
        lines += ["", "## Projected FCF", "| Year | FCF |", "|------|-----|"]
        for entry in dcf_projection:
            if isinstance(entry, dict):
                lines.append(f"| {entry.get('year', '?')} | {entry.get('fcf', 'N/A')} |")
    if dcf_sensitivity:
        lines += [
            "",
            "## Sensitivity Table (selected rows)",
            "| WACC | Terminal Growth | Intrinsic Value |",
            "|------|----------------|-----------------|",
        ]
        for entry in dcf_sensitivity[:10]:
            if isinstance(entry, dict):
                lines.append(
                    f"| {entry.get('wacc', '?')} | {entry.get('terminal_growth', '?')} | {entry.get('intrinsic_value', 'N/A')} |"
                )
    lines += [
        "",
        f"**Guidance Summary:** {r.get('DCF_Guidance_Summary') or 'N/A'}",
        f"**Filing Form:** {r.get('DCF_Filing_Form') or 'N/A'}  |  **Filing Date:** {r.get('DCF_Filing_Date') or 'N/A'}",
        "",
        "---",
        "<!-- Skill: /financial-analysis:dcf-model -->",
        "<!-- Usage: In Claude Code, run /financial-analysis:dcf-model and attach this file. The skill will build a full Excel DCF model from these assumptions. -->",
    ]
    return "\n".join(lines)


def build_ic_memo_skill_brief(record):
    """Build a Markdown input brief for ``/private-equity:ic-memo`` or ``/investment-banking:one-pager``.

    Parameters
    ----------
    record:
        A dict or pandas Series representing a saved analysis row.

    Returns
    -------
    str
        Markdown text, or an empty string when *record* is None.
    """
    if record is None:
        return ""
    r = record if isinstance(record, dict) else record.to_dict()
    ticker = normalize_ticker(r.get("Ticker"))

    lines = [
        f"# IC Memo / Investment Brief — {ticker}",
        "_Use as input for `/private-equity:ic-memo` or `/investment-banking:one-pager`._",
        "",
        "## Company Profile",
        f"- **Ticker:** {ticker}",
        f"- **Sector:** {r.get('Sector') or 'N/A'}  |  **Industry:** {r.get('Industry') or 'N/A'}",
        f"- **Stock Type:** {r.get('Stock_Type') or 'N/A'}  |  **Cap Bucket:** {r.get('Cap_Bucket') or 'N/A'}",
        f"- **Market Cap:** {_fmt(r.get('Market_Cap'), '${:,.0f}M')}",
        f"- **Current Price:** {_fmt(r.get('Price'), '${:.2f}')}",
        f"- **Style Tags:** {r.get('Style_Tags') or 'N/A'}",
        f"- **Type Strategy:** {r.get('Type_Strategy') or 'N/A'}",
        "",
        "## Financial Summary",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Revenue Growth (YoY) | {_fmt(r.get('Revenue_Growth'), '{:.1%}')} |",
        f"| Profit Margin | {_fmt(r.get('Profit_Margins'), '{:.1%}')} |",
        f"| ROE | {_fmt(r.get('ROE'), '{:.1%}')} |",
        f"| Debt/Equity | {_fmt(r.get('Debt_to_Equity'), '{:.2f}')} |",
        f"| Current Ratio | {_fmt(r.get('Current_Ratio'), '{:.2f}')} |",
        f"| Quality Score | {_fmt(r.get('Quality_Score'), '{:.1f}')} |",
        f"| Dividend Yield | {_fmt(r.get('Dividend_Yield'), '{:.2%}')} |",
        f"| Beta | {_fmt(r.get('Equity_Beta'), '{:.2f}')} |",
        "",
        "## Valuation",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Current Price | {_fmt(r.get('Price'), '${:.2f}')} |",
        f"| DCF Intrinsic Value | {_fmt(r.get('DCF_Intrinsic_Value'), '${:.2f}')} |",
        f"| DCF Upside | {_fmt(r.get('DCF_Upside'), '{:.1%}')} |",
        f"| Trailing P/E | {_fmt(r.get('PE_Ratio'), '{:.1f}x')} |",
        f"| Forward P/E | {_fmt(r.get('Forward_PE'), '{:.1f}x')} |",
        f"| EV/EBITDA | {_fmt(r.get('EV_EBITDA'), '{:.1f}x')} |",
        f"| P/S | {_fmt(r.get('PS_Ratio'), '{:.1f}x')} |",
        f"| Graham Number | {_fmt(r.get('Graham_Number'), '${:.2f}')} |",
        "",
        "## Investment Verdict",
        f"- **Overall:** {r.get('Verdict_Overall') or 'N/A'}",
        f"- **Confidence:** {_fmt(r.get('Decision_Confidence'), '{:.1%}')}",
        f"- **Notes:** {r.get('Decision_Notes') or 'N/A'}",
        f"- **Analyst Consensus:** {r.get('Recommendation_Key') or 'N/A'}  (Target: {_fmt(r.get('Target_Mean_Price'), '${:.2f}')})",
        "",
        "## Risk Factors",
        f"{r.get('Risk_Flags') or 'None identified.'}",
        "",
        "## Technical Context",
        f"- **Market Regime:** {r.get('Market_Regime') or 'N/A'}",
        f"- **RSI:** {_fmt(r.get('RSI'), '{:.1f}')}  |  **MACD:** {r.get('MACD_Signal') or 'N/A'}",
        f"- **Volatility 1Y:** {_fmt(r.get('Volatility_1Y'), '{:.1%}')}",
        f"- **52-Week Range Position:** {_fmt(r.get('Range_Position_52W'), '{:.1%}')}",
        "",
        "---",
        "<!-- Skill: /private-equity:ic-memo  OR  /investment-banking:one-pager -->",
        "<!-- Usage: In Claude Code, run the skill and attach this file when prompted for company data. -->",
    ]
    return "\n".join(lines)


def build_rebalance_skill_brief(portfolio_result, portfolio_name):
    """Build a Markdown input brief for the ``/wealth-management:rebalance`` skill.

    Parameters
    ----------
    portfolio_result:
        The dict returned by ``PortfolioAnalyst.analyze_portfolio()``.
    portfolio_name:
        A display name for the portfolio (used as a header).

    Returns
    -------
    str
        Markdown text, or an empty string when *portfolio_result* is falsy.
    """
    if not portfolio_result:
        return ""
    recommendations = portfolio_result.get("recommendations")
    asset_metrics = portfolio_result.get("asset_metrics")
    period = portfolio_result.get("period", "N/A")
    benchmark = portfolio_result.get("benchmark", "N/A")

    lines = [
        f"# Portfolio Rebalance Brief — {portfolio_name}",
        "_Use as input for `/wealth-management:rebalance`._",
        "",
        f"- **Portfolio:** {portfolio_name}",
        f"- **Benchmark:** {benchmark}  |  **Lookback Period:** {period}",
        "",
        "## Recommended Allocations (Efficient Frontier — Tangent Portfolio)",
    ]
    if recommendations is not None and not recommendations.empty and "Recommended Weight" in recommendations.columns:
        lines += [
            "",
            "| Ticker | Name | Sector | Recommended Weight | Role |",
            "|--------|------|--------|-------------------|------|",
        ]
        for rec in recommendations.itertuples(index=False):
            weight = getattr(rec, "Recommended Weight", "N/A")
            name = getattr(rec, "Name", "N/A")
            sector = getattr(rec, "Sector", "N/A")
            role = getattr(rec, "Role", "N/A")
            ticker = getattr(rec, "Ticker", "N/A")
            lines.append(f"| {ticker} | {name} | {sector} | {weight} | {role} |")

    if asset_metrics is not None and not asset_metrics.empty:
        lines += [
            "",
            "## Asset-Level Risk/Return Metrics",
            "| Ticker | Annual Return | Annual Volatility | Sharpe |",
            "|--------|--------------|-------------------|--------|",
        ]
        disp_cols = [c for c in ["Ticker", "Annual Return", "Annual Volatility", "Sharpe"] if c in asset_metrics.columns]
        if len(disp_cols) == 4:
            for row in asset_metrics[disp_cols].itertuples(index=False):
                lines.append(
                    f"| {row.Ticker} | {_fmt(getattr(row, 'Annual Return', None), '{:.1%}')} "
                    f"| {_fmt(getattr(row, 'Annual Volatility', None), '{:.1%}')} "
                    f"| {_fmt(getattr(row, 'Sharpe', None), '{:.2f}')} |"
                )
    lines += [
        "",
        "---",
        "<!-- Skill: /wealth-management:rebalance -->",
        "<!-- Usage: In Claude Code, run /wealth-management:rebalance and attach this file. -->",
        "<!-- The skill will analyze drift from targets and generate a trade list. -->",
        "<!-- Note: Add your current holdings and cost basis before running the skill. -->",
    ]
    return "\n".join(lines)
