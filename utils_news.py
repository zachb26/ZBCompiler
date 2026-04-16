# -*- coding: utf-8 -*-
"""
utils_news.py — News payload parsing helpers.

No streamlit imports. Depends only on utils_time.
"""

from utils_time import format_datetime_value, parse_any_datetime


def extract_news_publish_time(item):
    """Extract a publish timestamp from a single yfinance news item dict.

    Tries several candidate keys and recurses into a ``"content"`` sub-dict when
    the top level yields nothing.

    Returns a timezone-naive datetime, or None when nothing parseable is found.
    """
    if not isinstance(item, dict):
        return None
    for key in ["providerPublishTime", "publishTime", "published", "published_at"]:
        parsed = parse_any_datetime(item.get(key))
        if parsed is not None:
            return parsed
    content = item.get("content")
    if isinstance(content, dict):
        return extract_news_publish_time(content)
    return None


def extract_news_title(item):
    """Extract the headline string from a single yfinance news item dict.

    Recurses into a ``"content"`` sub-dict when the top-level ``"title"`` is empty.
    Returns an empty string when no title can be found.
    """
    if not isinstance(item, dict):
        return ""
    title = str(item.get("title") or "").strip()
    if title:
        return title
    content = item.get("content")
    if isinstance(content, dict):
        return str(content.get("title") or "").strip()
    return ""


def build_news_context_lines(news, max_items=5):
    """Build a list of formatted headline strings from a list of news item dicts.

    Each line is ``"YYYY-MM-DD | Publisher: Headline"`` (or a shortened variant
    when some fields are absent).  At most *max_items* lines are returned.
    """
    lines = []
    for item in news or []:
        title = extract_news_title(item)
        if not title:
            continue
        published = format_datetime_value(extract_news_publish_time(item), fallback="")
        publisher = ""
        if isinstance(item, dict):
            publisher = str(item.get("publisher") or item.get("source") or "").strip()
            if not publisher and isinstance(item.get("content"), dict):
                publisher = str(item["content"].get("publisher") or "").strip()
        prefix_parts = [part for part in [published, publisher] if part]
        prefix = " | ".join(prefix_parts)
        lines.append(f"{prefix}: {title}" if prefix else title)
        if len(lines) >= max_items:
            break
    return lines
