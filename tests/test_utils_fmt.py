"""tests/test_utils_fmt.py — Unit tests for utils_fmt pure helpers."""
import math
import pytest
import pandas as pd
from utils_fmt import (
    normalize_ticker,
    parse_ticker_list,
    safe_num,
    has_numeric_value,
    safe_divide,
    safe_json_loads,
    format_value,
    format_percent,
    format_int,
    format_market_cap,
    get_color,
    tone_to_color,
    escape_markdown_text,
    escape_html_text,
    colorize_markdown_text,
    calculate_rsi,
    score_relative_multiple,
    extract_sentiment_tokens,
)


# ---------------------------------------------------------------------------
# normalize_ticker
# ---------------------------------------------------------------------------
class TestNormalizeTicker:
    def test_uppercases(self):
        assert normalize_ticker("aapl") == "AAPL"

    def test_strips_whitespace(self):
        assert normalize_ticker("  msft  ") == "MSFT"

    def test_none_returns_empty(self):
        assert normalize_ticker(None) == ""

    def test_empty_string(self):
        assert normalize_ticker("") == ""


# ---------------------------------------------------------------------------
# parse_ticker_list
# ---------------------------------------------------------------------------
class TestParseTickerList:
    def test_comma_separated(self):
        assert parse_ticker_list("AAPL,MSFT,NVDA") == ["AAPL", "MSFT", "NVDA"]

    def test_space_separated(self):
        assert parse_ticker_list("AAPL MSFT NVDA") == ["AAPL", "MSFT", "NVDA"]

    def test_newline_separated(self):
        assert parse_ticker_list("AAPL\nMSFT\nNVDA") == ["AAPL", "MSFT", "NVDA"]

    def test_deduplicates(self):
        result = parse_ticker_list("AAPL, aapl, MSFT")
        assert result == ["AAPL", "MSFT"]

    def test_empty_string(self):
        assert parse_ticker_list("") == []


# ---------------------------------------------------------------------------
# safe_num
# ---------------------------------------------------------------------------
class TestSafeNum:
    def test_int(self):
        assert safe_num(5) == 5.0

    def test_float(self):
        assert safe_num(3.14) == pytest.approx(3.14)

    def test_numeric_string(self):
        assert safe_num("42.5") == pytest.approx(42.5)

    def test_percent_string(self):
        assert safe_num("12.5%") == pytest.approx(12.5)

    def test_comma_string(self):
        assert safe_num("1,000") == pytest.approx(1000.0)

    def test_na_string(self):
        assert safe_num("N/A") is None

    def test_none(self):
        assert safe_num(None) is None

    def test_nan_string(self):
        assert safe_num("nan") is None

    def test_non_numeric_string(self):
        assert safe_num("hello") is None


# ---------------------------------------------------------------------------
# has_numeric_value
# ---------------------------------------------------------------------------
class TestHasNumericValue:
    def test_real_number(self):
        assert has_numeric_value(1.5) is True

    def test_zero(self):
        assert has_numeric_value(0) is True

    def test_none(self):
        assert has_numeric_value(None) is False

    def test_nan(self):
        assert has_numeric_value(float("nan")) is False

    def test_pd_na(self):
        assert has_numeric_value(pd.NA) is False


# ---------------------------------------------------------------------------
# safe_divide
# ---------------------------------------------------------------------------
class TestSafeDivide:
    def test_basic(self):
        assert safe_divide(10, 2) == pytest.approx(5.0)

    def test_zero_denominator(self):
        assert safe_divide(10, 0) is None

    def test_none_denominator(self):
        assert safe_divide(10, None) is None

    def test_nan_denominator(self):
        assert safe_divide(10, float("nan")) is None

    def test_tiny_denominator(self):
        assert safe_divide(1, 1e-13) is None

    def test_negative(self):
        assert safe_divide(-6, 3) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# safe_json_loads
# ---------------------------------------------------------------------------
class TestSafeJsonLoads:
    def test_valid_json(self):
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_invalid_json_returns_default(self):
        assert safe_json_loads("not-json") == {}

    def test_none_returns_default(self):
        assert safe_json_loads(None) == {}

    def test_dict_passthrough(self):
        d = {"x": 2}
        result = safe_json_loads(d)
        assert result == d
        assert result is not d  # deep copy

    def test_custom_default(self):
        assert safe_json_loads(None, default={"k": 0}) == {"k": 0}


# ---------------------------------------------------------------------------
# format_value
# ---------------------------------------------------------------------------
class TestFormatValue:
    def test_basic(self):
        assert format_value(1234.5) == "1,234.50"

    def test_none_returns_na(self):
        assert format_value(None) == "N/A"

    def test_nan_returns_na(self):
        assert format_value(float("nan")) == "N/A"

    def test_suffix(self):
        assert format_value(10.0, suffix="x") == "10.00x"


# ---------------------------------------------------------------------------
# format_percent
# ---------------------------------------------------------------------------
class TestFormatPercent:
    def test_basic(self):
        assert format_percent(0.125) == "12.5%"

    def test_none(self):
        assert format_percent(None) == "N/A"

    def test_negative(self):
        assert format_percent(-0.05) == "-5.0%"


# ---------------------------------------------------------------------------
# format_int
# ---------------------------------------------------------------------------
class TestFormatInt:
    def test_basic(self):
        assert format_int(42.9) == "42"

    def test_none(self):
        assert format_int(None) == "N/A"


# ---------------------------------------------------------------------------
# format_market_cap
# ---------------------------------------------------------------------------
class TestFormatMarketCap:
    def test_trillions(self):
        assert format_market_cap(2_500_000_000_000) == "$2.50T"

    def test_billions(self):
        assert format_market_cap(1_500_000_000) == "$1.5B"

    def test_millions(self):
        assert format_market_cap(250_000_000) == "$250.0M"

    def test_small(self):
        assert format_market_cap(500_000) == "$500,000"

    def test_none(self):
        assert format_market_cap(None) == "N/A"


# ---------------------------------------------------------------------------
# get_color
# ---------------------------------------------------------------------------
class TestGetColor:
    def test_buy(self):
        assert get_color("BUY") == "green"

    def test_strong_buy(self):
        assert get_color("STRONG BUY") == "green"

    def test_sell(self):
        assert get_color("SELL") == "red"

    def test_hold(self):
        assert get_color("HOLD") == "gray"

    def test_undervalued(self):
        assert get_color("UNDERVALUED") == "green"

    def test_overvalued(self):
        assert get_color("OVERVALUED") == "red"


# ---------------------------------------------------------------------------
# tone_to_color
# ---------------------------------------------------------------------------
class TestToneToColor:
    def test_good(self):
        assert tone_to_color("good") == "green"

    def test_bad(self):
        assert tone_to_color("bad") == "red"

    def test_neutral(self):
        assert tone_to_color("neutral") == "gray"

    def test_unknown(self):
        assert tone_to_color("weird") == "gray"

    def test_none(self):
        assert tone_to_color(None) == "gray"


# ---------------------------------------------------------------------------
# escape_markdown_text
# ---------------------------------------------------------------------------
class TestEscapeMarkdownText:
    def test_brackets(self):
        assert escape_markdown_text("[link]") == "\\[link\\]"

    def test_backslash(self):
        assert escape_markdown_text("a\\b") == "a\\\\b"

    def test_plain(self):
        assert escape_markdown_text("hello") == "hello"


# ---------------------------------------------------------------------------
# colorize_markdown_text
# ---------------------------------------------------------------------------
class TestColorizeMarkdownText:
    def test_green(self):
        assert colorize_markdown_text("UP", "green") == ":green[UP]"

    def test_red(self):
        assert colorize_markdown_text("DOWN", "red") == ":red[DOWN]"

    def test_gray(self):
        assert colorize_markdown_text("FLAT", "gray") == ":gray[FLAT]"

    def test_unknown_color_returns_plain(self):
        assert colorize_markdown_text("X", "blue") == "X"


# ---------------------------------------------------------------------------
# escape_html_text
# ---------------------------------------------------------------------------
class TestEscapeHtmlText:
    def test_ampersand(self):
        assert escape_html_text("a & b") == "a &amp; b"

    def test_angle_brackets(self):
        assert escape_html_text("<div>") == "&lt;div&gt;"

    def test_quote(self):
        assert escape_html_text('"hello"') == "&quot;hello&quot;"

    def test_plain(self):
        assert escape_html_text("hello") == "hello"


# ---------------------------------------------------------------------------
# calculate_rsi
# ---------------------------------------------------------------------------
class TestCalculateRsi:
    def _trending_up(self, n=50):
        return pd.Series([float(100 + i) for i in range(n)])

    def _trending_down(self, n=50):
        return pd.Series([float(100 - i) for i in range(n)])

    def _flat(self, n=50):
        return pd.Series([100.0] * n)

    def test_overbought_on_strong_uptrend(self):
        rsi = calculate_rsi(self._trending_up())
        assert rsi.iloc[-1] > 70

    def test_oversold_on_strong_downtrend(self):
        rsi = calculate_rsi(self._trending_down())
        assert rsi.iloc[-1] < 30

    def test_flat_prices_return_50(self):
        rsi = calculate_rsi(self._flat())
        assert rsi.iloc[-1] == pytest.approx(50, abs=5)

    def test_rsi_bounded(self):
        rsi = calculate_rsi(self._trending_up())
        assert rsi.dropna().between(0, 100).all()

    def test_length_preserved(self):
        series = self._trending_up(30)
        rsi = calculate_rsi(series)
        assert len(rsi) == 30


# ---------------------------------------------------------------------------
# score_relative_multiple
# ---------------------------------------------------------------------------
class TestScoreRelativeMultiple:
    def test_cheap(self):
        assert score_relative_multiple(10, 15) == 1

    def test_expensive(self):
        assert score_relative_multiple(20, 15) == -1

    def test_neutral(self):
        assert score_relative_multiple(15, 15) == 0

    def test_none_value(self):
        assert score_relative_multiple(None, 15) == 0

    def test_zero_value(self):
        assert score_relative_multiple(0, 15) == -1

    def test_negative_value(self):
        assert score_relative_multiple(-5, 15) == -1

    def test_no_benchmark(self):
        assert score_relative_multiple(10, None) == 0


# ---------------------------------------------------------------------------
# extract_sentiment_tokens
# ---------------------------------------------------------------------------
class TestExtractSentimentTokens:
    def test_basic(self):
        assert extract_sentiment_tokens("Strong Buy") == {"strong", "buy"}

    def test_empty(self):
        assert extract_sentiment_tokens("") == set()

    def test_none(self):
        assert extract_sentiment_tokens(None) == set()

    def test_lowercase_words_extracted(self):
        # regex matches [a-z]+, so "Q3" → "q", "2024" → nothing, "earnings" → "earnings"
        result = extract_sentiment_tokens("Q3 2024 earnings")
        assert "earnings" in result
        assert "q" in result  # lowercase fragment of "Q3"
