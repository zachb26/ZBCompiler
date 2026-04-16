# -*- coding: utf-8 -*-
"""
dcf.py — Discounted Cash Flow valuation engine.

Functions:
    calculate_growth_rate_from_series
    build_growth_schedule
    fetch_treasury_10y_yield
    compute_wacc_components
    determine_growth_assumptions
    extract_recent_metric_values
    calculate_normalized_base_fcf
    estimate_terminal_exit_value
    build_dcf_sensitivity_grid
    build_sec_dcf_model
"""

import datetime

import numpy as np

from constants import (
    DCF_DEFAULT_AFTER_TAX_COST_OF_DEBT,
    DCF_DEFAULT_MARKET_RISK_PREMIUM,
    DCF_DEFAULT_RISK_FREE_RATE,
    DCF_TERMINAL_GROWTH_RATE,
)
from fetch import (
    has_numeric_value,
    safe_divide,
)
from utils_fmt import safe_num
from sec_ai import (
    fetch_sec_companyfacts,
    fetch_sec_submissions,
    fetch_sec_filing_text,
    parse_sec_filing_metadata,
    build_sec_financial_dataset,
    lookup_company_cik,
    fetch_json_url_with_retry,
)
from settings import normalize_dcf_settings


# ---------------------------------------------------------------------------
# Growth-rate helpers
# ---------------------------------------------------------------------------

def calculate_growth_rate_from_series(history_rows, field_name, lookback_years=3):
    """
    Compute a compound or mean annual growth rate for *field_name* across
    up to *lookback_years* of *history_rows*.

    Returns a decimal fraction or None.
    """
    valid_rows = [row for row in history_rows if has_numeric_value(row.get(field_name))]
    if len(valid_rows) < 2:
        return None

    end_row = valid_rows[-1]
    start_index = max(0, len(valid_rows) - lookback_years - 1)
    start_row = valid_rows[start_index]
    year_span = max(int(end_row["Year"]) - int(start_row["Year"]), 1)
    start_value = safe_num(start_row.get(field_name))
    end_value = safe_num(end_row.get(field_name))

    if has_numeric_value(start_value) and has_numeric_value(end_value) and start_value > 0 and end_value > 0:
        return float((end_value / start_value) ** (1 / year_span) - 1)

    pct_changes = []
    for previous_row, current_row in zip(valid_rows[:-1], valid_rows[1:]):
        previous_value = safe_num(previous_row.get(field_name))
        current_value = safe_num(current_row.get(field_name))
        if has_numeric_value(previous_value) and has_numeric_value(current_value) and abs(previous_value) > 1e-9:
            pct_changes.append((current_value - previous_value) / abs(previous_value))
    if pct_changes:
        return float(np.mean(pct_changes[-lookback_years:]))
    return None


def build_growth_schedule(initial_growth_rate, terminal_growth_rate, years):
    """Return a list of linearly interpolated annual growth rates."""
    if years <= 1:
        return [float(terminal_growth_rate)]
    return [float(value) for value in np.linspace(initial_growth_rate, terminal_growth_rate, years)]


# ---------------------------------------------------------------------------
# Treasury yield
# ---------------------------------------------------------------------------

def fetch_treasury_10y_yield():
    """
    Fetch the current 10-year US Treasury yield from the Treasury API.

    Tries the current and two prior months. Returns a decimal fraction
    (e.g. 0.045) and an error string or None.
    """
    now = datetime.datetime.now()
    month_candidates = [now, now - datetime.timedelta(days=35), now - datetime.timedelta(days=70)]
    last_error = None
    for candidate in month_candidates:
        cache_key = f"{candidate.year}-{candidate.month:02d}"
        from fetch import get_cached_fetch_payload, set_cached_fetch_payload
        cached = get_cached_fetch_payload("treasury_yield", cache_key, max_age_seconds=86400)
        if cached is not None:
            return float(cached), None

        url = (
            "https://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData"
            f"?$filter=month(NEW_DATE)%20eq%20{candidate.month}%20and%20year(NEW_DATE)%20eq%20{candidate.year}"
            "&$select=NEW_DATE,BC_10YEAR&$orderby=NEW_DATE%20desc&$format=json"
        )
        payload, error = fetch_json_url_with_retry(url, attempts=2, timeout=15)
        if error:
            last_error = error
        results = payload.get("d", {}).get("results", []) if isinstance(payload, dict) else []
        for item in results:
            rate = safe_num(item.get("BC_10YEAR"))
            if has_numeric_value(rate):
                decimal_rate = float(rate) / 100
                set_cached_fetch_payload("treasury_yield", cache_key, decimal_rate)
                return decimal_rate, None
    return DCF_DEFAULT_RISK_FREE_RATE, last_error or "Treasury yield fetch failed; used fallback."


# ---------------------------------------------------------------------------
# WACC
# ---------------------------------------------------------------------------

def compute_wacc_components(ticker, info, sec_dataset, dcf_settings):
    """
    Compute all WACC components from SEC data, company info, and DCF
    settings.

    Returns a dict with keys: risk_free_rate, beta, market_risk_premium,
    cost_of_equity, after_tax_cost_of_debt, equity_weight, debt_weight,
    wacc, cash, debt_balance, shares_outstanding, market_cap, tax_rate,
    notes.
    """
    latest = sec_dataset.get("latest", {})
    manual_risk_free_rate = safe_num((dcf_settings or {}).get("risk_free_rate_override"))
    if has_numeric_value(manual_risk_free_rate):
        risk_free_rate = float(manual_risk_free_rate)
        rf_error = None
    else:
        risk_free_rate, rf_error = fetch_treasury_10y_yield()
    beta = safe_num((info or {}).get("beta"))
    if not has_numeric_value(beta):
        beta = 1.0

    market_cap = safe_num((info or {}).get("marketCap"))
    debt_balance = safe_num(latest.get("DebtBalance"))
    if not has_numeric_value(debt_balance):
        debt_balance = safe_num((info or {}).get("totalDebt"))
    cash = safe_num(latest.get("Cash"))
    shares_outstanding = safe_num(latest.get("SharesOutstanding")) or safe_num((info or {}).get("sharesOutstanding"))

    market_risk_premium = float((dcf_settings or {}).get("market_risk_premium", DCF_DEFAULT_MARKET_RISK_PREMIUM))
    cost_of_equity = float(risk_free_rate + beta * market_risk_premium)

    pretax_income = safe_num(latest.get("PretaxIncome"))
    tax_expense = safe_num(latest.get("TaxExpense"))
    effective_tax_rate = safe_divide(tax_expense, pretax_income)
    if not has_numeric_value(effective_tax_rate):
        effective_tax_rate = 0.21
    effective_tax_rate = float(np.clip(effective_tax_rate, 0.0, 0.45))

    interest_expense = safe_num(latest.get("InterestExpense"))
    pre_tax_cost_of_debt = (
        safe_divide(abs(interest_expense), abs(debt_balance))
        if has_numeric_value(interest_expense) and has_numeric_value(debt_balance)
        else None
    )
    if not has_numeric_value(pre_tax_cost_of_debt) or pre_tax_cost_of_debt <= 0 or pre_tax_cost_of_debt > 0.20:
        after_tax_cost_of_debt = float(
            (dcf_settings or {}).get("default_after_tax_cost_of_debt", DCF_DEFAULT_AFTER_TAX_COST_OF_DEBT)
        )
    else:
        after_tax_cost_of_debt = float(pre_tax_cost_of_debt * (1 - effective_tax_rate))

    if has_numeric_value(market_cap) and has_numeric_value(debt_balance) and (market_cap + debt_balance) > 0:
        equity_weight = float(market_cap / (market_cap + debt_balance))
        debt_weight = float(debt_balance / (market_cap + debt_balance))
    elif has_numeric_value(market_cap) and market_cap > 0:
        equity_weight = 1.0
        debt_weight = 0.0
    elif has_numeric_value(debt_balance) and debt_balance > 0:
        equity_weight = 0.0
        debt_weight = 1.0
    else:
        equity_weight = 1.0
        debt_weight = 0.0

    wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)
    wacc = float(np.clip(wacc, 0.06, 0.20))

    return {
        "risk_free_rate": float(risk_free_rate),
        "beta": float(beta),
        "market_risk_premium": market_risk_premium,
        "cost_of_equity": cost_of_equity,
        "after_tax_cost_of_debt": after_tax_cost_of_debt,
        "equity_weight": equity_weight,
        "debt_weight": debt_weight,
        "wacc": wacc,
        "cash": cash,
        "debt_balance": debt_balance,
        "shares_outstanding": shares_outstanding,
        "market_cap": market_cap,
        "tax_rate": effective_tax_rate,
        "notes": [rf_error] if rf_error else [],
    }


# ---------------------------------------------------------------------------
# Growth assumptions
# ---------------------------------------------------------------------------

def determine_growth_assumptions(history_rows, dcf_settings=None):
    """
    Derive the growth schedule for the DCF projection from SEC history and
    optional DCF settings overrides.

    Returns a dict with keys: historical_fcf_growth, historical_revenue_growth,
    historical_growth_estimate, guidance_rate, selected_growth_rate, source,
    confidence, summary, growth_schedule.
    """
    dcf_settings = normalize_dcf_settings(dcf_settings or {})
    historical_fcf_growth = calculate_growth_rate_from_series(history_rows, "FreeCashFlow", lookback_years=3)
    historical_revenue_growth = calculate_growth_rate_from_series(history_rows, "Revenue", lookback_years=3)
    if has_numeric_value(historical_fcf_growth) and has_numeric_value(historical_revenue_growth):
        growth_gap = abs(float(historical_fcf_growth) - float(historical_revenue_growth))
        if growth_gap >= 0.12:
            historical_growth_estimate = float(
                historical_fcf_growth * 0.35 + historical_revenue_growth * 0.65
            )
        else:
            historical_growth_estimate = float(np.mean([historical_fcf_growth, historical_revenue_growth]))
    elif has_numeric_value(historical_revenue_growth):
        historical_growth_estimate = float(historical_revenue_growth)
    elif has_numeric_value(historical_fcf_growth):
        historical_growth_estimate = float(historical_fcf_growth)
    else:
        historical_growth_estimate = 0.04

    manual_growth_rate = safe_num(dcf_settings.get("manual_growth_rate"))
    selected_source_rate = manual_growth_rate if has_numeric_value(manual_growth_rate) else historical_growth_estimate
    guidance_source = "Manual override" if has_numeric_value(manual_growth_rate) else "Historical trend fallback"
    guidance_confidence = "manual" if has_numeric_value(manual_growth_rate) else "historical"
    guidance_summary = (
        "Used the manual growth override supplied in the DCF tab."
        if has_numeric_value(manual_growth_rate)
        else "Used recent SEC cash-flow and revenue history as the base growth assumption."
    )

    selected_pre_haircut = selected_source_rate
    selected_growth_rate = float(
        np.clip(
            selected_pre_haircut * dcf_settings["growth_haircut"],
            dcf_settings["min_growth_rate"],
            dcf_settings["max_growth_rate"],
        )
    )

    return {
        "historical_fcf_growth": historical_fcf_growth,
        "historical_revenue_growth": historical_revenue_growth,
        "historical_growth_estimate": historical_growth_estimate,
        "guidance_rate": manual_growth_rate,
        "selected_growth_rate": selected_growth_rate,
        "source": guidance_source,
        "confidence": guidance_confidence,
        "summary": guidance_summary,
        "growth_schedule": build_growth_schedule(
            selected_growth_rate,
            dcf_settings["terminal_growth_rate"],
            dcf_settings["projection_years"],
        ),
    }


# ---------------------------------------------------------------------------
# FCF normalisation
# ---------------------------------------------------------------------------

def extract_recent_metric_values(history_rows, field_name, limit=5):
    """Return the last *limit* non-null float values for *field_name*."""
    values = []
    for row in history_rows or []:
        value = safe_num(row.get(field_name))
        if has_numeric_value(value):
            values.append(float(value))
    return values[-limit:]


def calculate_normalized_base_fcf(history_rows, latest_metrics=None):
    """
    Derive a normalised base FCF that smooths out capex spikes and
    negative interim periods.

    Returns a dict with keys: base_fcf, latest_fcf, normalized_fcf,
    maintenance_capex, used_normalized_capex, capex_source.
    """
    latest_metrics = latest_metrics or {}
    valid_rows = [
        row for row in history_rows or []
        if has_numeric_value(row.get("OperatingCF")) or has_numeric_value(row.get("FreeCashFlow"))
    ]
    if not valid_rows:
        return {
            "base_fcf": None,
            "latest_fcf": None,
            "normalized_fcf": None,
            "maintenance_capex": None,
            "used_normalized_capex": False,
            "capex_source": "Unavailable",
        }

    latest_row = valid_rows[-1]
    latest_operating_cf = safe_num(latest_row.get("OperatingCF"))
    latest_capex = safe_num(latest_row.get("CapEx"))
    latest_fcf = safe_num(latest_row.get("FreeCashFlow"))
    depreciation = safe_num(latest_metrics.get("Depreciation"))

    recent_capex_values = extract_recent_metric_values(valid_rows, "CapEx", limit=5)
    recent_fcf_values = extract_recent_metric_values(valid_rows, "FreeCashFlow", limit=5)
    recent_capex_window = recent_capex_values[-3:] if recent_capex_values else []
    recent_fcf_window = recent_fcf_values[-3:] if recent_fcf_values else []
    median_recent_capex = float(np.median(recent_capex_window)) if recent_capex_window else None
    median_recent_fcf = float(np.median(recent_fcf_window)) if recent_fcf_window else None

    maintenance_floor = depreciation if has_numeric_value(depreciation) else median_recent_capex
    maintenance_capex = latest_capex
    capex_source = "Latest reported capex"
    used_normalized_capex = False

    if has_numeric_value(median_recent_capex):
        spike_threshold = max(
            median_recent_capex * 1.35,
            (maintenance_floor if has_numeric_value(maintenance_floor) else median_recent_capex) * 1.75,
        )
        if not has_numeric_value(maintenance_capex):
            maintenance_capex = max(
                median_recent_capex,
                maintenance_floor if has_numeric_value(maintenance_floor) else 0.0,
            )
            capex_source = "Recent median capex"
            used_normalized_capex = True
        elif maintenance_capex > spike_threshold:
            maintenance_capex = max(
                median_recent_capex,
                maintenance_floor if has_numeric_value(maintenance_floor) else 0.0,
            )
            capex_source = "Normalized capex from recent median"
            used_normalized_capex = True
    elif not has_numeric_value(maintenance_capex) and has_numeric_value(maintenance_floor):
        maintenance_capex = float(maintenance_floor)
        capex_source = "Depreciation proxy"
        used_normalized_capex = True

    normalized_fcf = (
        latest_operating_cf - maintenance_capex
        if has_numeric_value(latest_operating_cf) and has_numeric_value(maintenance_capex)
        else latest_fcf
    )

    candidate_fcfs = []
    if has_numeric_value(normalized_fcf):
        candidate_fcfs.append(float(normalized_fcf))
    if has_numeric_value(median_recent_fcf):
        candidate_fcfs.append(float(median_recent_fcf))
    if len(recent_fcf_window) >= 2:
        weights = np.arange(1, len(recent_fcf_window) + 1)
        candidate_fcfs.append(float(np.average(recent_fcf_window, weights=weights)))

    positive_recent_count = sum(value > 0 for value in recent_fcf_window)
    if (
        has_numeric_value(latest_fcf)
        and latest_fcf > 0
        and positive_recent_count >= 2
        and not used_normalized_capex
    ):
        candidate_fcfs.append(float(latest_fcf))

    if not candidate_fcfs:
        base_fcf = latest_fcf
    elif positive_recent_count == 0 and (not has_numeric_value(normalized_fcf) or normalized_fcf <= 0):
        base_fcf = max(candidate_fcfs)
    else:
        positive_candidates = [value for value in candidate_fcfs if value > 0]
        base_fcf = float(np.mean(positive_candidates or candidate_fcfs))

    return {
        "base_fcf": base_fcf,
        "latest_fcf": latest_fcf,
        "normalized_fcf": normalized_fcf,
        "maintenance_capex": maintenance_capex,
        "used_normalized_capex": used_normalized_capex,
        "capex_source": capex_source,
    }


# ---------------------------------------------------------------------------
# Terminal exit value
# ---------------------------------------------------------------------------

def estimate_terminal_exit_value(info, sec_dataset, growth_schedule, wacc, projection_years, peer_benchmarks=None):
    """
    Estimate a blended terminal value using EV/EBITDA and EV/Sales multiples.

    Returns a dict with keys: terminal_value, present_value, methods,
    projected_revenue, projected_ebitda, or {} if insufficient data.
    """
    latest = sec_dataset.get("latest", {})
    latest_revenue = safe_num(latest.get("Revenue"))
    if not has_numeric_value(latest_revenue) or latest_revenue <= 0:
        return {}

    projected_revenue = float(latest_revenue)
    for growth_rate in growth_schedule or []:
        projected_revenue *= (1 + growth_rate)

    current_ebitda = safe_num((info or {}).get("ebitda"))
    if not has_numeric_value(current_ebitda):
        operating_income = safe_num(latest.get("OperatingIncome"))
        depreciation = safe_num(latest.get("Depreciation"))
        amortization = safe_num(latest.get("Amortization"))
        if has_numeric_value(operating_income):
            current_ebitda = float(operating_income + (depreciation or 0.0) + (amortization or 0.0))

    exit_candidates = []
    year5_ebitda = None
    current_ev_ebitda = safe_num((info or {}).get("enterpriseToEbitda"))
    peer_ev_ebitda = safe_num((peer_benchmarks or {}).get("EV_EBITDA"))
    ebitda_multiple_candidates = []
    if has_numeric_value(current_ev_ebitda) and current_ev_ebitda > 0:
        ebitda_multiple_candidates.append(float(np.clip(current_ev_ebitda * 0.90, 6.0, 20.0)))
    if has_numeric_value(peer_ev_ebitda) and peer_ev_ebitda > 0:
        ebitda_multiple_candidates.append(float(np.clip(peer_ev_ebitda, 6.0, 18.0)))

    if has_numeric_value(current_ebitda) and current_ebitda > 0 and has_numeric_value(latest_revenue):
        ebitda_margin = float(current_ebitda / latest_revenue)
        if 0.02 <= ebitda_margin <= 0.65 and ebitda_multiple_candidates:
            year5_ebitda = float(projected_revenue * ebitda_margin)
            exit_candidates.append(
                {
                    "method": "EV/EBITDA",
                    "terminal_value": year5_ebitda * float(np.mean(ebitda_multiple_candidates)),
                }
            )

    current_ev_revenue = safe_num((info or {}).get("enterpriseToRevenue"))
    if has_numeric_value(current_ev_revenue) and current_ev_revenue > 0:
        revenue_multiple = float(np.clip(current_ev_revenue * 0.85, 1.0, 12.0))
        exit_candidates.append(
            {
                "method": "EV/Sales",
                "terminal_value": projected_revenue * revenue_multiple,
            }
        )

    if not exit_candidates:
        return {}

    terminal_value = float(np.mean([candidate["terminal_value"] for candidate in exit_candidates]))
    present_value = terminal_value / ((1 + wacc) ** projection_years)
    return {
        "terminal_value": terminal_value,
        "present_value": present_value,
        "methods": [candidate["method"] for candidate in exit_candidates],
        "projected_revenue": projected_revenue,
        "projected_ebitda": year5_ebitda,
    }


# ---------------------------------------------------------------------------
# Sensitivity grid
# ---------------------------------------------------------------------------

def build_dcf_sensitivity_grid(projected_fcfs, wacc, cash, debt, shares_outstanding, dcf_settings):
    """
    Build a sensitivity table of per-share intrinsic values across a range of
    WACC and terminal-growth-rate combinations.

    Returns a list of row dicts.
    """
    sensitivity_rows = []
    if not projected_fcfs or not has_numeric_value(shares_outstanding) or shares_outstanding <= 0:
        return sensitivity_rows

    wacc_range = [round(wacc + delta, 4) for delta in np.arange(-0.02, 0.0201, 0.005)]
    center_terminal_growth = float((dcf_settings or {}).get("terminal_growth_rate", DCF_TERMINAL_GROWTH_RATE))
    terminal_growth_range = sorted(
        {
            round(max(0.0, center_terminal_growth - 0.01), 3),
            round(max(0.0, center_terminal_growth - 0.005), 3),
            round(center_terminal_growth, 3),
            round(center_terminal_growth + 0.005, 3),
            round(center_terminal_growth + 0.01, 3),
        }
    )

    for wacc_candidate in wacc_range:
        row = {"WACC": wacc_candidate}
        for growth_candidate in terminal_growth_range:
            if wacc_candidate <= growth_candidate + 0.0025:
                row[f"TG_{growth_candidate:.3f}"] = None
                continue
            pv_fcfs = sum(
                projected_fcf / ((1 + wacc_candidate) ** year)
                for year, projected_fcf in enumerate(projected_fcfs, start=1)
            )
            terminal_value = projected_fcfs[-1] * (1 + growth_candidate) / (wacc_candidate - growth_candidate)
            pv_terminal = terminal_value / ((1 + wacc_candidate) ** len(projected_fcfs))
            enterprise_value = pv_fcfs + pv_terminal
            equity_value = enterprise_value - (debt or 0) + (cash or 0)
            row[f"TG_{growth_candidate:.3f}"] = safe_divide(equity_value, shares_outstanding)
        sensitivity_rows.append(row)
    return sensitivity_rows


# ---------------------------------------------------------------------------
# Full DCF model entry point
# ---------------------------------------------------------------------------

def build_sec_dcf_model(ticker, price, info, dcf_settings=None, peer_benchmarks=None):
    """
    Build a complete DCF valuation model sourced from SEC EDGAR filings.

    Returns a dict.  On failure ``available`` is False and ``error`` is set.
    On success ``available`` is True and the dict contains projection,
    sensitivity, WACC components, intrinsic value, and guidance data.
    """
    from fetch import extract_filing_takeaways_from_text

    dcf_settings = normalize_dcf_settings(dcf_settings or {})
    cik_padded, company_title, cik_error = lookup_company_cik(ticker)
    if not cik_padded:
        return {"available": False, "error": cik_error or f"Could not find an SEC CIK for {ticker}."}

    companyfacts, facts_error = fetch_sec_companyfacts(cik_padded)
    if not companyfacts:
        return {
            "available": False,
            "error": facts_error or f"SEC company facts were unavailable for {ticker}.",
            "cik": cik_padded,
        }

    submissions, submissions_error = fetch_sec_submissions(cik_padded)
    sec_dataset = build_sec_financial_dataset(companyfacts)
    history_rows = sec_dataset.get("history", [])
    latest = sec_dataset.get("latest", {})

    base_fcf_inputs = calculate_normalized_base_fcf(history_rows, latest_metrics=latest)
    base_fcf = safe_num(base_fcf_inputs.get("base_fcf"))
    if not has_numeric_value(base_fcf):
        return {
            "available": False,
            "error": "SEC filing history did not provide enough operating cash flow and capex data to build a DCF.",
            "cik": cik_padded,
            "company_name": companyfacts.get("entityName") or company_title or ticker,
        }

    filing_metadata = parse_sec_filing_metadata(submissions or {}, preferred_forms=SEC_FILING_SEARCH_FORMS)
    filing_text, filing_text_error = (
        fetch_sec_filing_text(cik_padded, filing_metadata) if filing_metadata else (None, submissions_error)
    )
    filing_takeaways = extract_filing_takeaways_from_text(filing_text) if filing_text else []
    growth_inputs = determine_growth_assumptions(history_rows, dcf_settings=dcf_settings)
    if base_fcf_inputs.get("used_normalized_capex"):
        growth_inputs["summary"] += (
            f" Base FCF was normalized using {base_fcf_inputs.get('capex_source', 'recent capex history')}."
        )

    wacc_inputs = compute_wacc_components(ticker, info, sec_dataset, dcf_settings=dcf_settings)
    shares_outstanding = safe_num(wacc_inputs.get("shares_outstanding")) or safe_num((info or {}).get("sharesOutstanding"))
    if not has_numeric_value(shares_outstanding) or shares_outstanding <= 0:
        return {
            "available": False,
            "error": "Shares outstanding were unavailable, so the DCF could not be converted into a per-share value.",
            "cik": cik_padded,
            "company_name": companyfacts.get("entityName") or company_title or ticker,
        }

    current_fcf = float(base_fcf)
    projected_fcfs = []
    projection_rows = []
    for year, growth_rate in enumerate(growth_inputs["growth_schedule"], start=1):
        current_fcf = current_fcf * (1 + growth_rate)
        discount_factor = 1 / ((1 + wacc_inputs["wacc"]) ** year)
        pv = current_fcf * discount_factor
        projected_fcfs.append(current_fcf)
        projection_rows.append(
            {
                "Year": year,
                "GrowthRate": growth_rate,
                "FreeCashFlow": current_fcf,
                "DiscountFactor": discount_factor,
                "PresentValue": pv,
            }
        )

    if wacc_inputs["wacc"] <= dcf_settings["terminal_growth_rate"] + 0.0025:
        return {
            "available": False,
            "error": "The DCF could not be stabilized because the discount rate was too close to terminal growth.",
            "cik": cik_padded,
            "company_name": companyfacts.get("entityName") or company_title or ticker,
        }

    terminal_value = projected_fcfs[-1] * (1 + dcf_settings["terminal_growth_rate"]) / (
        wacc_inputs["wacc"] - dcf_settings["terminal_growth_rate"]
    )
    present_value_of_terminal = terminal_value / ((1 + wacc_inputs["wacc"]) ** dcf_settings["projection_years"])
    exit_terminal = estimate_terminal_exit_value(
        info,
        sec_dataset,
        growth_inputs["growth_schedule"],
        wacc_inputs["wacc"],
        dcf_settings["projection_years"],
        peer_benchmarks=peer_benchmarks,
    )
    if has_numeric_value(exit_terminal.get("present_value")):
        exit_weight = 0.35
        if (
            base_fcf_inputs.get("used_normalized_capex")
            and has_numeric_value(base_fcf_inputs.get("latest_fcf"))
            and base_fcf_inputs["latest_fcf"] <= 0
        ):
            exit_weight = 0.50
        present_value_of_terminal = (
            present_value_of_terminal * (1 - exit_weight)
            + exit_terminal["present_value"] * exit_weight
        )
        terminal_value = terminal_value * (1 - exit_weight) + exit_terminal["terminal_value"] * exit_weight
    sum_of_projected_pv = float(sum(row["PresentValue"] for row in projection_rows))
    enterprise_value = sum_of_projected_pv + present_value_of_terminal
    equity_value = enterprise_value - (wacc_inputs.get("debt_balance") or 0) + (wacc_inputs.get("cash") or 0)
    intrinsic_value_per_share = safe_divide(equity_value, shares_outstanding)
    dcf_upside = safe_divide(intrinsic_value_per_share - price, price) if has_numeric_value(price) else None

    notes = []
    if cik_error:
        notes.append(cik_error)
    if facts_error:
        notes.append(facts_error)
    if submissions_error:
        notes.append(submissions_error)
    if filing_text_error:
        notes.append(filing_text_error)
    notes.extend(wacc_inputs.get("notes", []))
    if base_fcf_inputs.get("used_normalized_capex"):
        notes.append(
            f"Base FCF normalized from recent operating cash flow and {base_fcf_inputs.get('capex_source', 'capex history')}."
        )
    if has_numeric_value(exit_terminal.get("present_value")):
        notes.append(
            f"Terminal value blended with {', '.join(exit_terminal.get('methods', []))} cross-check."
        )

    return {
        "available": True,
        "ticker": ticker,
        "company_name": companyfacts.get("entityName") or company_title or ticker,
        "cik": cik_padded,
        "filing_form": filing_metadata.get("form") if filing_metadata else None,
        "filing_date": filing_metadata.get("filing_date") if filing_metadata else None,
        "history": history_rows,
        "projection": projection_rows,
        "sensitivity": build_dcf_sensitivity_grid(
            projected_fcfs,
            wacc_inputs["wacc"],
            wacc_inputs.get("cash"),
            wacc_inputs.get("debt_balance"),
            shares_outstanding,
            dcf_settings,
        ),
        "guidance_excerpts": filing_takeaways,
        "guidance_summary": growth_inputs["summary"],
        "guidance_payload": {},
        "historical_fcf_growth": growth_inputs["historical_fcf_growth"],
        "historical_revenue_growth": growth_inputs["historical_revenue_growth"],
        "guidance_growth_rate": growth_inputs["guidance_rate"],
        "selected_growth_rate": growth_inputs["selected_growth_rate"],
        "growth_source": growth_inputs["source"],
        "growth_confidence": growth_inputs["confidence"],
        "growth_schedule": growth_inputs["growth_schedule"],
        "base_fcf": base_fcf,
        "terminal_growth_rate": dcf_settings["terminal_growth_rate"],
        "risk_free_rate": wacc_inputs["risk_free_rate"],
        "beta": wacc_inputs["beta"],
        "cost_of_equity": wacc_inputs["cost_of_equity"],
        "after_tax_cost_of_debt": wacc_inputs["after_tax_cost_of_debt"],
        "equity_weight": wacc_inputs["equity_weight"],
        "debt_weight": wacc_inputs["debt_weight"],
        "wacc": wacc_inputs["wacc"],
        "cash": wacc_inputs.get("cash"),
        "long_term_debt": wacc_inputs.get("debt_balance"),
        "shares_outstanding": shares_outstanding,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "sum_of_projected_pv": sum_of_projected_pv,
        "terminal_value": terminal_value,
        "present_value_of_terminal": present_value_of_terminal,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "upside": dcf_upside,
        "notes": [note for note in notes if note],
        "latest_sec_values": latest,
        "dcf_settings": dcf_settings,
    }
