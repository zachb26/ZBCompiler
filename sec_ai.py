# -*- coding: utf-8 -*-
"""
sec_ai.py — SEC EDGAR API client, AI guidance extraction, and in-app AI
report generation.

Public functions / constants:
    get_sec_request_headers
    get_sec_access_hint
    explain_upstream_fetch_error
    fetch_json_url_with_retry
    fetch_text_url_with_retry
    load_sec_ticker_map
    lookup_company_cik
    fetch_sec_submissions
    fetch_sec_companyfacts
    parse_sec_filing_metadata
    fetch_sec_filing_text
    strip_html_to_text
    extract_guidance_excerpts_from_text
    extract_json_object_from_text
    extract_guidance_with_anthropic
    SKILL_REPORT_TYPES
    call_claude_for_skill_report
    render_ai_reports_tab          ← only function that imports streamlit
    parse_percentage_range
    extract_regex_guidance
    parse_year_from_date
    sec_entry_priority
    extract_company_fact_entries
    latest_sec_metric_value
    build_sec_financial_dataset
"""

import logging
import os
import re
import time

import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd

from constants import (
    SEC_ANNUAL_FORMS,
    SEC_EDGAR_CONTACT_EMAIL,
    SEC_EDGAR_ORGANIZATION,
    SEC_FILING_SEARCH_FORMS,
    SEC_GUIDANCE_PATTERNS,
    SEC_REQUEST_DELAY_SECONDS,
    SEC_USER_AGENT,
)
from fetch import (
    get_cached_fetch_payload,
    has_numeric_value,
    safe_json_loads,
    set_cached_fetch_payload,
    summarize_fetch_error,
)
from utils_fmt import normalize_ticker, safe_num


# ---------------------------------------------------------------------------
# Headers and hints
# ---------------------------------------------------------------------------

def get_sec_request_headers():
    """Return the HTTP headers required for SEC EDGAR API access."""
    user_agent = SEC_USER_AGENT or (
        f"{SEC_EDGAR_ORGANIZATION} {SEC_EDGAR_CONTACT_EMAIL}"
        if SEC_EDGAR_CONTACT_EMAIL
        else f"{SEC_EDGAR_ORGANIZATION} (set SEC_EDGAR_CONTACT_EMAIL for SEC EDGAR access)"
    )
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.8",
    }
    if SEC_EDGAR_CONTACT_EMAIL:
        headers["From"] = SEC_EDGAR_CONTACT_EMAIL
    return headers


def get_sec_access_hint():
    """Return a user-facing hint for fixing SEC EDGAR 403 errors."""
    return (
        "Set SEC_EDGAR_CONTACT_EMAIL=you@example.com before starting the app. "
        "You can also set SEC_EDGAR_ORGANIZATION='Your App Name' or provide a full SEC_EDGAR_USER_AGENT value."
    )


def explain_upstream_fetch_error(url, exc):
    """Return a human-readable error message, with special handling for HTTP 403."""
    message = summarize_fetch_error(exc)
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if "sec.gov" in str(url).lower() and (status_code == 403 or "403" in message):
        return f"SEC EDGAR returned HTTP 403. {get_sec_access_hint()}"
    return message


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch_json_url_with_retry(url, *, headers=None, attempts=3, timeout=15):
    """
    Fetch a JSON payload from *url* with retry.

    Returns (payload_or_None, error_string_or_None).
    """
    try:
        import requests
    except ImportError:
        return None, "The requests library is not installed, so SEC and Treasury data could not be fetched."

    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            time.sleep(SEC_REQUEST_DELAY_SECONDS)
            return payload, None
        except Exception as exc:
            last_error = explain_upstream_fetch_error(url, exc)
            logger.warning("fetch_json_url attempt %d failed (%s): %s", attempt + 1, url, exc)
            if "SEC EDGAR returned HTTP 403" in str(last_error):
                break
        if attempt < attempts - 1:
            time.sleep(0.35 * (attempt + 1))
    return None, last_error


def fetch_text_url_with_retry(url, *, headers=None, attempts=3, timeout=20):
    """
    Fetch raw text from *url* with retry.

    Returns (text_or_None, error_string_or_None).
    """
    try:
        import requests
    except ImportError:
        return None, "The requests library is not installed, so filing text could not be fetched."

    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            time.sleep(SEC_REQUEST_DELAY_SECONDS)
            return response.text, None
        except Exception as exc:
            last_error = explain_upstream_fetch_error(url, exc)
            logger.warning("fetch_text_url attempt %d failed (%s): %s", attempt + 1, url, exc)
            if "SEC EDGAR returned HTTP 403" in str(last_error):
                break
        if attempt < attempts - 1:
            time.sleep(0.35 * (attempt + 1))
    return None, last_error


# ---------------------------------------------------------------------------
# SEC EDGAR API
# ---------------------------------------------------------------------------

def load_sec_ticker_map():
    """
    Download and normalise the SEC company-tickers JSON, caching the result.

    Returns (ticker_map_dict, error_or_None).
    """
    cached = get_cached_fetch_payload("sec_ticker_map", "normalized")
    if cached:
        return cached, None

    payload, error = fetch_json_url_with_retry(
        "https://www.sec.gov/files/company_tickers.json",
        headers=get_sec_request_headers(),
        attempts=2,
        timeout=15,
    )
    if not isinstance(payload, dict):
        return {}, error

    normalized = {}
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        ticker = normalize_ticker(item.get("ticker", ""))
        cik_value = item.get("cik_str")
        if not ticker or cik_value in {None, ""}:
            continue
        normalized[ticker] = {
            "cik": str(int(cik_value)).zfill(10),
            "title": str(item.get("title", ticker)).strip(),
        }

    if normalized:
        set_cached_fetch_payload("sec_ticker_map", "normalized", normalized)
    return normalized, error


def lookup_company_cik(ticker):
    """Return (cik_padded, company_title, error) for the given *ticker*."""
    ticker_map, error = load_sec_ticker_map()
    payload = ticker_map.get(normalize_ticker(ticker))
    if payload:
        return payload["cik"], payload.get("title"), None
    return None, None, error or f"SEC ticker mapping did not contain {ticker}."


def fetch_sec_submissions(cik_padded):
    """
    Fetch the SEC submissions JSON for the company identified by *cik_padded*.

    Returns (submissions_dict_or_None, error_or_None).
    """
    cached = get_cached_fetch_payload("sec_submissions", cik_padded)
    if cached:
        return cached, None

    payload, error = fetch_json_url_with_retry(
        f"https://data.sec.gov/submissions/CIK{cik_padded}.json",
        headers=get_sec_request_headers(),
        attempts=2,
        timeout=15,
    )
    if isinstance(payload, dict):
        set_cached_fetch_payload("sec_submissions", cik_padded, payload)
        return payload, None
    return None, error


def fetch_sec_companyfacts(cik_padded):
    """
    Fetch the SEC company-facts XBRL JSON for the company identified by
    *cik_padded*.

    Returns (companyfacts_dict_or_None, error_or_None).
    """
    cached = get_cached_fetch_payload("sec_companyfacts", cik_padded)
    if cached:
        return cached, None

    payload, error = fetch_json_url_with_retry(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json",
        headers=get_sec_request_headers(),
        attempts=2,
        timeout=20,
    )
    if isinstance(payload, dict):
        set_cached_fetch_payload("sec_companyfacts", cik_padded, payload)
        return payload, None
    return None, error


def parse_sec_filing_metadata(submissions, preferred_forms=None):
    """
    Walk the recent-filings index in *submissions* and return metadata for
    the first matching form.

    Returns a dict with keys form, accession_number, primary_document,
    filing_date, or None if no match is found.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = list(recent.get("form", []) or [])
    accession_numbers = list(recent.get("accessionNumber", []) or [])
    primary_docs = list(recent.get("primaryDocument", []) or [])
    filing_dates = list(recent.get("filingDate", []) or [])

    for target_form in preferred_forms or SEC_FILING_SEARCH_FORMS:
        for idx, form in enumerate(forms):
            if form != target_form:
                continue
            accession_number = accession_numbers[idx] if idx < len(accession_numbers) else None
            primary_doc = primary_docs[idx] if idx < len(primary_docs) else None
            filing_date = filing_dates[idx] if idx < len(filing_dates) else None
            if accession_number and primary_doc:
                return {
                    "form": form,
                    "accession_number": accession_number,
                    "primary_document": primary_doc,
                    "filing_date": filing_date,
                }
    return None


def fetch_sec_filing_text(cik_padded, filing_metadata):
    """
    Fetch the primary filing document text for the given *filing_metadata*.

    Returns (text_or_None, error_or_None).
    """
    if not filing_metadata:
        return None, "No recent 10-K or 10-Q filing metadata was available."

    accession_number = filing_metadata.get("accession_number", "")
    accession_no_dashes = accession_number.replace("-", "")
    primary_document = filing_metadata.get("primary_document", "")
    cik_numeric = str(int(cik_padded))
    cache_key = (cik_padded, accession_no_dashes, primary_document)
    cached = get_cached_fetch_payload("sec_filing_text", cache_key)
    if cached:
        return cached, None

    filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{accession_no_dashes}/{primary_document}"
    payload, error = fetch_text_url_with_retry(
        filing_url,
        headers=get_sec_request_headers(),
        attempts=2,
        timeout=20,
    )
    if payload:
        set_cached_fetch_payload("sec_filing_text", cache_key, payload)
        return payload, None
    return None, error


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def strip_html_to_text(html_text):
    """Return plain text stripped of HTML tags, using BeautifulSoup when available."""
    if not html_text:
        return ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_text, "html.parser")
        text = soup.get_text(" ")
    except Exception:
        text = re.sub(r"<[^>]+>", " ", str(html_text))
    return " ".join(str(text).split())


def extract_guidance_excerpts_from_text(filing_text, *, max_excerpts=3, window_sentences=4):
    """
    Extract up to *max_excerpts* forward-guidance excerpts from raw filing
    text by searching for SEC guidance patterns and returning a surrounding
    sentence window.
    """
    clean_text = strip_html_to_text(filing_text)
    if not clean_text:
        return []

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", clean_text) if sentence.strip()]
    excerpts = []
    seen = set()
    for idx, sentence in enumerate(sentences):
        lowered = sentence.lower()
        if not any(pattern in lowered for pattern in SEC_GUIDANCE_PATTERNS):
            continue
        start = max(0, idx - window_sentences)
        end = min(len(sentences), idx + window_sentences + 1)
        excerpt = " ".join(sentences[start:end]).strip()
        excerpt = " ".join(excerpt.split())
        if len(excerpt) < 80:
            continue
        dedupe_key = excerpt.lower()[:320]
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        excerpts.append(excerpt)
        if len(excerpts) >= max_excerpts:
            break
    return excerpts


def extract_json_object_from_text(text):
    """Extract the first JSON object found in *text*."""
    if not text:
        return None
    match = re.search(r"\{.*\}", str(text), flags=re.DOTALL)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# AI guidance extraction via Anthropic
# ---------------------------------------------------------------------------

def extract_guidance_with_anthropic(excerpts):
    """
    Call the Claude API to parse structured guidance data from filing
    *excerpts*.

    Returns (guidance_dict_or_None, error_or_None).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not excerpts:
        return None, None

    try:
        from anthropic import Anthropic
    except Exception:
        return None, "Anthropic guidance extraction was skipped because the anthropic package is unavailable."

    prompt = (
        "You are a financial analyst. From these excerpts of an SEC filing, extract any specific "
        "numerical forward guidance the company provides, including expected revenue growth %, "
        "earnings growth %, margin targets, or any other quantitative forward-looking statements. "
        "Return a JSON object with keys: revenue_growth_pct, earnings_growth_pct, margin_target_pct, "
        "other_guidance (list of strings), confidence (low/medium/high). "
        "If a value is not found, return null for that key.\n\n"
        "Filing excerpts:\n"
        + "\n\n".join(f"{idx + 1}. {excerpt}" for idx, excerpt in enumerate(excerpts))
    )

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = ""
        for block in getattr(response, "content", []):
            if getattr(block, "type", "") == "text":
                response_text += getattr(block, "text", "")
        json_payload = extract_json_object_from_text(response_text)
        parsed = safe_json_loads(json_payload, default={}) if json_payload else {}
        if isinstance(parsed, dict) and parsed:
            return parsed, None
    except Exception as exc:
        logger.error("extract_guidance_with_anthropic failed: %s", exc)
        return None, summarize_fetch_error(exc)
    return None, "Anthropic guidance extraction returned no usable JSON payload."


# ---------------------------------------------------------------------------
# AI Reports — skill-mirroring Claude API calls for in-app report generation
# ---------------------------------------------------------------------------

SKILL_REPORT_TYPES = {
    "Earnings Analysis": {
        "skill": "/equity-research:earnings",
        "brief_fn": "earnings",
        "description": "Institutional earnings update: beat/miss analysis, key metrics, updated estimates, revised thesis.",
        "system": (
            "You are an institutional equity research analyst at a top-tier investment bank. "
            "Produce a professional earnings update report in markdown. Structure it with: "
            "1) Executive Summary (rating, price target if derivable, 1-paragraph verdict), "
            "2) Beat/Miss Analysis (revenue, margins, key metrics vs. model expectations), "
            "3) Investment Thesis Update (bull/base/bear case impact), "
            "4) Valuation Commentary (multiple context vs. peers), "
            "5) Key Risks. "
            "Use only the data provided. Do not invent numbers. Be precise and concise."
        ),
    },
    "Investment Thesis": {
        "skill": "/equity-research:thesis",
        "brief_fn": "ic_memo",
        "description": "Bull/base/bear thesis with catalysts, risks, and price target rationale.",
        "system": (
            "You are a senior equity research analyst. Produce a structured investment thesis in markdown. "
            "Include: 1) One-line verdict, 2) Bull case (3 pillars with supporting data), "
            "3) Bear case (3 risks with mitigants), 4) Base case valuation (DCF-anchored if data present), "
            "5) Key catalysts (3-5 events that could move the stock), 6) Recommendation summary. "
            "Use only the data provided. Do not invent numbers."
        ),
    },
    "Comparable Company Analysis": {
        "skill": "/financial-analysis:comps-analysis",
        "brief_fn": "comps",
        "description": "Peer valuation table with median/quartile benchmarks and relative positioning.",
        "system": (
            "You are a financial analyst building a comparable company analysis. "
            "From the provided data, produce a markdown comps summary that includes: "
            "1) Comps table (EV/EBITDA, P/E, P/S, Revenue Growth, Margin for subject + peers), "
            "2) Statistical summary (max, 75th pct, median, 25th pct, min for each multiple), "
            "3) Relative positioning narrative (where the subject trades vs. peers and why), "
            "4) Key takeaway. Use only the data provided."
        ),
    },
    "Deal Screening Memo": {
        "skill": "/private-equity:screen-deal",
        "brief_fn": "ic_memo",
        "description": "One-page pass / further-diligence / hard-pass verdict with bull and bear cases.",
        "system": (
            "You are a private equity associate screening an inbound deal. "
            "Produce a one-page screening memo in markdown with: "
            "1) Deal Facts (company, sector, revenue, EBITDA/margins, growth, valuation implied), "
            "2) Fund Fit (does it match typical PE criteria: size, sector, growth profile?), "
            "3) Verdict: Pass / Further Diligence / Hard Pass, "
            "4) Bull Case (2-3 bullets: why it could be attractive), "
            "5) Bear Case (2-3 bullets: key risks/red flags), "
            "6) Key Questions for first call. "
            "Use only the data provided. Do not invent numbers."
        ),
    },
}


def call_claude_for_skill_report(report_key, record):
    """
    Call the Claude API to generate a skill-quality markdown report from the
    analysis *record* dict.

    Raises ValueError / RuntimeError when the API key is missing or the
    anthropic package is unavailable.
    """
    from skill_briefs import (
        build_comps_skill_brief,
        build_earnings_skill_brief,
        build_ic_memo_skill_brief,
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your environment to use AI Reports.")
    try:
        from anthropic import Anthropic
    except Exception as exc:
        raise RuntimeError("The anthropic package is unavailable.") from exc

    config = SKILL_REPORT_TYPES[report_key]
    brief_fn_name = config["brief_fn"]
    if brief_fn_name == "earnings":
        brief = build_earnings_skill_brief(record)
    elif brief_fn_name == "comps":
        brief = build_comps_skill_brief(record)
    elif brief_fn_name == "ic_memo":
        brief = build_ic_memo_skill_brief(record)
    else:
        brief = build_earnings_skill_brief(record)

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=config["system"],
        messages=[{"role": "user", "content": brief}],
    )
    return "".join(
        getattr(block, "text", "") for block in getattr(response, "content", [])
        if getattr(block, "type", "") == "text"
    )


def render_ai_reports_tab(db):
    """
    Render the AI Reports sub-tab within the Analyst section.

    Requires streamlit — only called from view modules.
    """
    import streamlit as st
    from analysis_prep import prepare_analysis_dataframe
    from utils_fmt import normalize_ticker

    st.caption(
        "Generate institutional-quality markdown reports directly from your saved analyses. "
        "Powered by Claude — requires `ANTHROPIC_API_KEY` in your environment."
    )
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not has_api_key:
        st.warning(
            "No `ANTHROPIC_API_KEY` found in the environment. "
            "Set it to enable report generation. The app's DCF guidance extraction uses the same key."
        )

    ticker_input = st.text_input(
        "Ticker to report on",
        value=st.session_state.get("new_analyst_ticker", ""),
        placeholder="e.g. AAPL",
        key="ai_reports_ticker",
    )
    ticker_input = normalize_ticker(ticker_input)

    report_key = st.selectbox(
        "Report type",
        list(SKILL_REPORT_TYPES.keys()),
        key="ai_reports_type",
    )
    config = SKILL_REPORT_TYPES[report_key]
    st.info(f"**Mirrors:** `{config['skill']}`  \n{config['description']}")

    generate_btn = st.button(
        "Generate Report",
        disabled=not has_api_key or not ticker_input,
        type="primary",
        key="ai_reports_generate",
    )

    result_key = f"ai_report_{ticker_input}_{report_key}"
    if generate_btn and ticker_input:
        analysis_df = prepare_analysis_dataframe(db.get_analysis(ticker_input))
        if analysis_df.empty:
            st.error(f"No saved analysis found for {ticker_input}. Run an analysis first in the New Analyst tab.")
        else:
            record = analysis_df.iloc[0].to_dict()
            with st.spinner("Generating report — typically 20-60 seconds..."):
                try:
                    report_text = call_claude_for_skill_report(report_key, record)
                    st.session_state[result_key] = report_text
                except Exception as exc:
                    logger.error("AI report generation failed (report_key=%s): %s", report_key, exc)
                    st.error(f"Report generation failed: {exc}")

    if st.session_state.get(result_key):
        report_text = st.session_state[result_key]
        st.markdown("---")
        st.markdown(report_text)
        st.download_button(
            "Download Report (.md)",
            data=report_text.encode("utf-8"),
            file_name=f"{ticker_input}_{report_key.lower().replace(' ', '_')}.md",
            mime="text/markdown",
            key=f"ai_reports_download_{result_key}",
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Guidance parsing helpers
# ---------------------------------------------------------------------------

def parse_percentage_range(text):
    """
    Extract a percentage from *text*, returning the midpoint of a range when
    found.  Returns a decimal fraction (e.g. 0.07 for 7%).
    """
    if not text:
        return None

    range_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:to|-|\u2013)\s*(\d+(?:\.\d+)?)\s*(?:%|percent)",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        low = safe_num(range_match.group(1))
        high = safe_num(range_match.group(2))
        if has_numeric_value(low) and has_numeric_value(high):
            return (float(low) + float(high)) / 2 / 100

    single_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", text, flags=re.IGNORECASE)
    if single_match:
        single_value = safe_num(single_match.group(1))
        if has_numeric_value(single_value):
            return float(single_value) / 100
    return None


def extract_regex_guidance(excerpts):
    """
    Extract structured guidance from *excerpts* using regex when the
    Anthropic API is unavailable.

    Returns a guidance dict or {} when nothing is found.
    """
    if not excerpts:
        return {}

    result = {
        "revenue_growth_pct": None,
        "earnings_growth_pct": None,
        "margin_target_pct": None,
        "other_guidance": [],
        "confidence": "low",
    }
    for excerpt in excerpts:
        lowered = excerpt.lower()
        pct = parse_percentage_range(excerpt)
        if pct is None:
            continue
        if "revenue" in lowered and result["revenue_growth_pct"] is None:
            result["revenue_growth_pct"] = pct * 100
        elif any(term in lowered for term in ["earnings", "eps", "profit"]) and result["earnings_growth_pct"] is None:
            result["earnings_growth_pct"] = pct * 100
        elif "margin" in lowered and result["margin_target_pct"] is None:
            result["margin_target_pct"] = pct * 100
        else:
            result["other_guidance"].append(excerpt[:220])
    if any(result[key] is not None for key in ["revenue_growth_pct", "earnings_growth_pct", "margin_target_pct"]):
        return result
    return {}


# ---------------------------------------------------------------------------
# XBRL fact helpers
# ---------------------------------------------------------------------------

def parse_year_from_date(value):
    """Extract a 4-digit year integer from an ISO date string."""
    if not value:
        return None
    try:
        return int(str(value)[:4])
    except ValueError:
        return None


def sec_entry_priority(entry):
    """Return a sort key that prefers full-year (FY) annual-form entries."""
    return (
        1 if str(entry.get("fp", "")).upper() == "FY" else 0,
        1 if str(entry.get("form", "")) in SEC_ANNUAL_FORMS else 0,
        str(entry.get("filed", "")),
        str(entry.get("end", "")),
    )


def extract_company_fact_entries(companyfacts, concepts, *, preferred_units=None, forms=None):
    """
    Walk the XBRL company-facts for any of *concepts* and return the
    de-duplicated, year-sorted list of the best-quality entries found.
    """
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    preferred_units = preferred_units or ["USD"]
    allowed_forms = set(forms or SEC_ANNUAL_FORMS)
    best_entries = []
    best_score = None

    for concept in concepts:
        concept_payload = facts.get(concept)
        if not isinstance(concept_payload, dict):
            continue

        unit_entries = concept_payload.get("units", {})
        entries = None
        for unit_name in preferred_units:
            if unit_name in unit_entries:
                entries = unit_entries[unit_name]
                break
        if entries is None and unit_entries:
            entries = next(iter(unit_entries.values()))
        if not entries:
            continue

        normalized = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            form = str(item.get("form", "")).strip()
            if allowed_forms and form not in allowed_forms:
                continue
            value = safe_num(item.get("val"))
            if value is None:
                continue
            year = item.get("fy")
            year = int(year) if str(year).isdigit() else parse_year_from_date(item.get("end"))
            if year is None:
                continue
            normalized.append(
                {
                    "concept": concept,
                    "value": float(value),
                    "year": year,
                    "end": item.get("end"),
                    "filed": item.get("filed"),
                    "form": form,
                    "fp": item.get("fp"),
                }
            )

        if not normalized:
            continue

        deduped = {}
        for entry in normalized:
            current = deduped.get(entry["year"])
            if current is None or sec_entry_priority(entry) > sec_entry_priority(current):
                deduped[entry["year"]] = entry

        selected_entries = [deduped[year] for year in sorted(deduped)]
        latest_year = max(entry["year"] for entry in selected_entries)
        nonzero_count = sum(abs(entry["value"]) > 1e-9 for entry in selected_entries)
        fy_count = sum(str(entry.get("fp", "")).upper() == "FY" for entry in selected_entries)
        latest_value = safe_num(selected_entries[-1].get("value"))
        concept_score = (
            latest_year,
            len(selected_entries),
            nonzero_count,
            fy_count,
            1 if has_numeric_value(latest_value) and abs(latest_value) > 1e-9 else 0,
        )
        if best_score is None or concept_score > best_score:
            best_entries = selected_entries
            best_score = concept_score

    return best_entries


def latest_sec_metric_value(entries):
    """Return the value field of the last entry in *entries*, or None."""
    if not entries:
        return None
    return entries[-1].get("value")


def build_sec_financial_dataset(companyfacts):
    """
    Extract a structured financial dataset from *companyfacts*, covering
    Revenue, OperatingCF, CapEx, NetIncome, and related metrics for the last
    5 fiscal years.

    Returns a dict with keys: history (list of year-dicts), latest (metric→value),
    metric_entries (metric→entries list).
    """
    metric_config = {
        "Revenue": {
            "concepts": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
            "preferred_units": ["USD"],
        },
        "OperatingCF": {
            "concepts": [
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            ],
            "preferred_units": ["USD"],
        },
        "CapEx": {
            "concepts": [
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "CapitalExpendituresIncurredButNotYetPaid",
                "PropertyPlantAndEquipmentAdditions",
            ],
            "preferred_units": ["USD"],
            "absolute_value": True,
        },
        "NetIncome": {"concepts": ["NetIncomeLoss"], "preferred_units": ["USD"]},
        "OperatingIncome": {"concepts": ["OperatingIncomeLoss"], "preferred_units": ["USD"]},
        "DebtBalance": {
            "concepts": [
                "DebtLongtermAndShorttermCombinedAmount",
                "LongTermDebtAndCapitalLeaseObligations",
                "LongTermDebtNoncurrent",
                "LongTermDebt",
            ],
            "preferred_units": ["USD"],
        },
        "Cash": {"concepts": ["CashAndCashEquivalentsAtCarryingValue"], "preferred_units": ["USD"]},
        "SharesOutstanding": {
            "concepts": ["CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
            "preferred_units": ["shares"],
        },
        "Depreciation": {
            "concepts": [
                "DepreciationDepletionAndAmortization",
                "DepreciationAndAmortization",
                "Depreciation",
            ],
            "preferred_units": ["USD"],
        },
        "Amortization": {
            "concepts": [
                "AmortizationOfIntangibleAssets",
                "FiniteLivedIntangibleAssetsAmortizationExpense",
            ],
            "preferred_units": ["USD"],
        },
        "TaxExpense": {"concepts": ["IncomeTaxExpenseBenefit"], "preferred_units": ["USD"]},
        "InterestExpense": {"concepts": ["InterestExpenseAndDebtExpense", "InterestExpense"], "preferred_units": ["USD"]},
        "PretaxIncome": {
            "concepts": [
                "IncomeBeforeTaxExpenseBenefit",
                "PretaxIncome",
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
            ],
            "preferred_units": ["USD"],
        },
    }

    metric_entries = {}
    year_map = {}
    for metric_name, config in metric_config.items():
        entries = extract_company_fact_entries(
            companyfacts,
            config["concepts"],
            preferred_units=config.get("preferred_units"),
            forms=SEC_ANNUAL_FORMS,
        )
        if config.get("absolute_value"):
            for entry in entries:
                entry["value"] = abs(entry["value"])
        metric_entries[metric_name] = entries
        for entry in entries:
            year_map.setdefault(entry["year"], {})[metric_name] = entry["value"]

    history_years = sorted(year_map)[-5:]
    history_rows = []
    for year in history_years:
        operating_cf = year_map[year].get("OperatingCF")
        capex = year_map[year].get("CapEx")
        free_cash_flow = (
            operating_cf - capex
            if has_numeric_value(operating_cf) and has_numeric_value(capex)
            else None
        )
        history_rows.append(
            {
                "Year": year,
                "Revenue": year_map[year].get("Revenue"),
                "OperatingCF": operating_cf,
                "CapEx": capex,
                "FreeCashFlow": free_cash_flow,
            }
        )

    latest = {metric_name: latest_sec_metric_value(entries) for metric_name, entries in metric_entries.items()}
    return {
        "history": history_rows,
        "latest": latest,
        "metric_entries": metric_entries,
    }
