# -*- coding: utf-8 -*-
"""
utils_ui.py — Streamlit UI helper components.

These are small, reusable Streamlit rendering helpers that are shared across
multiple view modules.  Streamlit IS imported here because these functions
render directly into the page.

Depends on: utils_fmt (for colorize_markdown_text, tone_to_color,
score_relative_multiple, has_numeric_value).
"""

import streamlit as st

from utils_fmt import (
    colorize_markdown_text,
    has_numeric_value,
    score_relative_multiple,
    tone_to_color,
)


def render_help_legend(items):
    """Render a compact row of captioned legend items with ? tooltips.

    Parameters
    ----------
    items:
        Iterable of ``(label, help_text)`` pairs.  Entries whose
        *help_text* is falsy are silently skipped.
    """
    legend_items = [(label, help_text) for label, help_text in items if help_text]
    if not legend_items:
        return

    st.caption("Hover the labels below for quick definitions.")
    legend_columns = st.columns(min(4, len(legend_items)))
    for idx, (label, help_text) in enumerate(legend_items):
        with legend_columns[idx % len(legend_columns)]:
            st.caption(label, help=help_text)


def tone_from_metric_threshold(value, *, good_min=None, good_max=None, bad_min=None, bad_max=None):
    """Derive a tone string (good / bad / neutral) by comparing *value* to thresholds.

    At least one threshold keyword must be supplied.  Thresholds are inclusive.
    """
    if not has_numeric_value(value):
        return "neutral"
    value = float(value)
    has_good_rule = good_min is not None or good_max is not None
    has_bad_rule = bad_min is not None or bad_max is not None

    is_good = True
    if good_min is not None and value < good_min:
        is_good = False
    if good_max is not None and value > good_max:
        is_good = False
    if has_good_rule and is_good:
        return "good"

    is_bad = True
    if bad_min is not None and value < bad_min:
        is_bad = False
    if bad_max is not None and value > bad_max:
        is_bad = False
    if has_bad_rule and is_bad:
        return "bad"

    return "neutral"


def tone_from_balanced_band(value, healthy_min, healthy_max, caution_low, caution_high):
    """Return a tone based on whether *value* falls inside a healthy band or outside a caution band.

    * ``"good"``    — *healthy_min* <= value <= *healthy_max*
    * ``"bad"``     — value <= *caution_low* or value >= *caution_high*
    * ``"neutral"`` — in between
    """
    if not has_numeric_value(value):
        return "neutral"
    value = float(value)
    if healthy_min <= value <= healthy_max:
        return "good"
    if value <= caution_low or value >= caution_high:
        return "bad"
    return "neutral"


def tone_from_signal_text(value, positives=None, negatives=None):
    """Return a tone by looking up the upper-cased *value* in positive/negative sets.

    Parameters
    ----------
    value:
        A text signal string (e.g. ``"HIGH"``, ``"BULLISH"``).
    positives:
        Set of strings that map to ``"good"``.
    negatives:
        Set of strings that map to ``"bad"``.
    """
    normalized = str(value or "").strip().upper()
    positive_values = set(positives or [])
    negative_values = set(negatives or [])
    if normalized in positive_values:
        return "good"
    if normalized in negative_values:
        return "bad"
    return "neutral"


def tone_from_quality_label(value):
    """Map a quality label string (High / Medium / Low) to a tone."""
    normalized = str(value or "").strip().title()
    if normalized == "High":
        return "good"
    if normalized == "Low":
        return "bad"
    return "neutral"


def tone_from_regime(value):
    """Map a market-regime string to a tone."""
    normalized = str(value or "").strip()
    if normalized == "Bullish Trend":
        return "good"
    if normalized == "Bearish Trend":
        return "bad"
    return "neutral"


def tone_from_relative_multiple(value, benchmark):
    """Return a tone derived from :func:`~utils_fmt.score_relative_multiple`."""
    score = score_relative_multiple(value, benchmark)
    if score > 0:
        return "good"
    if score < 0:
        return "bad"
    return "neutral"


def render_analysis_signal_cards(items, columns=4):
    """Render a row of bordered metric cards with tone-coloured values.

    Parameters
    ----------
    items:
        List of dicts with keys: ``label``, ``value``, ``tone``, ``note``
        (optional), ``help`` (optional).
    columns:
        Number of columns in the grid.
    """
    if not items:
        return

    cols = st.columns(columns)
    for idx, item in enumerate(items):
        tone = item.get("tone", "neutral")
        label = str(item.get("label", ""))
        value = str(item.get("value", ""))
        note = str(item.get("note", "")).strip()
        badge_label = "Constructive" if tone == "good" else "Caution" if tone == "bad" else "Mixed"
        with cols[idx % columns]:
            with st.container(border=True):
                st.caption(label, help=item.get("help"))
                st.markdown(f"### {colorize_markdown_text(value, tone_to_color(tone))}")
                st.badge(badge_label, color=tone_to_color(tone))
                if note:
                    st.caption(note)


def render_analysis_signal_table(rows, reference_label="Reference"):
    """Render a tabular comparison of metric / value / reference / read.

    Parameters
    ----------
    rows:
        List of dicts with keys: ``metric``, ``value``, ``reference``,
        ``tone``, ``status`` (optional), ``help`` (optional).
    reference_label:
        Header text for the reference column.
    """
    if not rows:
        return

    with st.container(border=True):
        header_cols = st.columns([1.4, 0.9, 0.9, 0.8])
        header_cols[0].caption("Metric")
        header_cols[1].caption("Value")
        header_cols[2].caption(str(reference_label))
        header_cols[3].caption("Read")

        for row in rows:
            st.divider()
            tone = row.get("tone", "neutral")
            metric = str(row.get("metric", ""))
            value = str(row.get("value", ""))
            reference = str(row.get("reference", ""))
            status = str(row.get("status", tone.title()))
            row_cols = st.columns([1.4, 0.9, 0.9, 0.8])
            row_cols[0].caption(metric, help=row.get("help"))
            row_cols[1].markdown(colorize_markdown_text(value, tone_to_color(tone)))
            row_cols[2].write(reference)
            row_cols[3].badge(status, color=tone_to_color(tone))
