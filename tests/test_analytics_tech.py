"""tests/test_analytics_tech.py — Unit tests for analytics_tech computations."""
import pytest
import numpy as np
import pandas as pd
from analytics_tech import (
    calculate_realized_volatility,
    calculate_trend_strength,
    calculate_52w_context,
    calculate_quality_score,
    calculate_dividend_safety_score,
    compute_relative_strength,
)

# Minimal settings dict matching the thresholds used in quality score
SETTINGS = {
    "fund_roe_threshold": 0.15,
    "fund_profit_margin_threshold": 0.10,
    "fund_debt_good_threshold": 1.0,
    "fund_debt_bad_threshold": 2.0,
    "fund_current_ratio_good": 1.5,
    "fund_current_ratio_bad": 1.0,
}


# ---------------------------------------------------------------------------
# calculate_realized_volatility
# ---------------------------------------------------------------------------
class TestRealizedVolatility:
    def _flat_series(self, n=60):
        return pd.Series([100.0] * n)

    def _volatile_series(self, n=60):
        rng = np.random.default_rng(42)
        return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, n)))

    def test_flat_prices_near_zero_vol(self):
        vol = calculate_realized_volatility(self._flat_series(), window=30)
        assert vol is not None
        assert vol == pytest.approx(0.0, abs=1e-6)

    def test_volatile_series_positive(self):
        vol = calculate_realized_volatility(self._volatile_series(), window=30)
        assert vol is not None
        assert vol > 0

    def test_annualised_scaling(self):
        # Daily std of ~2 % → annualised ~0.02 * sqrt(252) ≈ 0.317
        vol = calculate_realized_volatility(self._volatile_series(), window=30)
        assert 0.05 < vol < 1.0

    def test_none_on_short_series(self):
        short = pd.Series([100.0, 101.0, 102.0])
        assert calculate_realized_volatility(short, window=30) is None

    def test_none_input(self):
        assert calculate_realized_volatility(None, window=30) is None


# ---------------------------------------------------------------------------
# calculate_trend_strength
# ---------------------------------------------------------------------------
class TestTrendStrength:
    def test_bullish_all_inputs(self):
        score = calculate_trend_strength(price=220, sma50=210, sma200=200, momentum_1y=0.25)
        assert score is not None
        assert score > 0

    def test_bearish_all_inputs(self):
        score = calculate_trend_strength(price=180, sma50=185, sma200=200, momentum_1y=-0.20)
        assert score is not None
        assert score < 0

    def test_bounded_positive(self):
        score = calculate_trend_strength(price=999, sma50=900, sma200=100, momentum_1y=10.0)
        assert score <= 100

    def test_bounded_negative(self):
        score = calculate_trend_strength(price=1, sma50=10, sma200=200, momentum_1y=-5.0)
        assert score >= -100

    def test_none_when_no_inputs(self):
        assert calculate_trend_strength(price=None, sma50=None, sma200=None) is None

    def test_price_only(self):
        # sma200 required for price component; without it, returns None
        assert calculate_trend_strength(price=100, sma50=None, sma200=None) is None

    def test_price_and_sma200(self):
        score = calculate_trend_strength(price=110, sma50=None, sma200=100)
        assert score is not None
        assert score > 0


# ---------------------------------------------------------------------------
# calculate_52w_context
# ---------------------------------------------------------------------------
class TestContext52w:
    def _series(self, low=80.0, high=120.0, last=100.0, n=252):
        data = [low] + [high] * (n - 2) + [last]
        return pd.Series(data, dtype=float)

    def test_midpoint_range_position(self):
        # price == midpoint of [80, 120] → range_position ≈ 0.5
        pos, _, _ = calculate_52w_context(self._series(low=80, high=120, last=100))
        assert pos == pytest.approx(0.5, abs=0.01)

    def test_at_52w_high(self):
        pos, dist_high, _ = calculate_52w_context(self._series(low=80, high=120, last=120))
        assert pos == pytest.approx(1.0, abs=0.01)
        assert dist_high == pytest.approx(0.0, abs=0.01)

    def test_at_52w_low(self):
        pos, _, dist_low = calculate_52w_context(self._series(low=80, high=120, last=80))
        assert pos == pytest.approx(0.0, abs=0.01)
        assert dist_low == pytest.approx(0.0, abs=0.01)

    def test_none_on_empty(self):
        assert calculate_52w_context(pd.Series([], dtype=float)) == (None, None, None)

    def test_none_on_none_input(self):
        assert calculate_52w_context(None) == (None, None, None)

    def test_distance_high_negative(self):
        # price below the 52w high → distance_high should be negative
        _, dist_high, _ = calculate_52w_context(self._series(low=80, high=120, last=100))
        assert dist_high < 0


# ---------------------------------------------------------------------------
# calculate_quality_score
# ---------------------------------------------------------------------------
class TestQualityScore:
    def test_high_quality_company(self):
        score = calculate_quality_score(
            roe=0.20,
            margins=0.15,
            debt_eq=0.5,
            revenue_growth=0.10,
            earnings_growth=0.12,
            current_ratio=2.0,
            settings=SETTINGS,
        )
        assert score == pytest.approx(5.0)  # max is 5

    def test_poor_quality_company(self):
        score = calculate_quality_score(
            roe=-0.05,
            margins=-0.02,
            debt_eq=3.0,
            revenue_growth=-0.10,
            earnings_growth=-0.15,
            current_ratio=0.8,
            settings=SETTINGS,
        )
        assert score == pytest.approx(-4.0)  # min is -4

    def test_missing_inputs_zero(self):
        score = calculate_quality_score(
            roe=None, margins=None, debt_eq=None,
            revenue_growth=None, earnings_growth=None,
            current_ratio=None, settings=SETTINGS,
        )
        assert score == pytest.approx(0.0)

    def test_bounded_above(self):
        score = calculate_quality_score(
            roe=0.99, margins=0.99, debt_eq=0.0,
            revenue_growth=1.0, earnings_growth=1.0,
            current_ratio=10.0, settings=SETTINGS,
        )
        assert score <= 5.0

    def test_bounded_below(self):
        score = calculate_quality_score(
            roe=-1.0, margins=-1.0, debt_eq=99.0,
            revenue_growth=-1.0, earnings_growth=-1.0,
            current_ratio=0.01, settings=SETTINGS,
        )
        assert score >= -4.0


# ---------------------------------------------------------------------------
# calculate_dividend_safety_score
# ---------------------------------------------------------------------------
class TestDividendSafetyScore:
    def test_safe_dividend(self):
        score = calculate_dividend_safety_score(
            dividend_yield=0.035,
            payout_ratio=0.50,
            margins=0.15,
            current_ratio=1.5,
            debt_eq=80,
        )
        assert score > 0

    def test_unsafe_dividend(self):
        score = calculate_dividend_safety_score(
            dividend_yield=0.10,
            payout_ratio=1.20,
            margins=-0.05,
            current_ratio=0.8,
            debt_eq=250,
        )
        assert score < 0

    def test_bounded_above(self):
        score = calculate_dividend_safety_score(
            dividend_yield=0.035, payout_ratio=0.30, margins=0.20,
            current_ratio=3.0, debt_eq=10,
        )
        assert score <= 4.0

    def test_bounded_below(self):
        score = calculate_dividend_safety_score(
            dividend_yield=0.10, payout_ratio=1.5, margins=-0.2,
            current_ratio=0.5, debt_eq=300,
        )
        assert score >= -3.0

    def test_none_inputs_zero(self):
        score = calculate_dividend_safety_score(
            dividend_yield=None, payout_ratio=None, margins=None,
            current_ratio=None, debt_eq=None,
        )
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_relative_strength
# ---------------------------------------------------------------------------
class TestComputeRelativeStrength:
    def _make_series(self, returns, start=100.0):
        prices = [start]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        return pd.Series(prices)

    def test_outperforming_stock(self):
        stock = self._make_series([0.01] * 50)
        bench = self._make_series([0.005] * 50)
        rs = compute_relative_strength(stock, bench, window=20)
        assert rs is not None
        assert rs > 0

    def test_underperforming_stock(self):
        stock = self._make_series([-0.005] * 50)
        bench = self._make_series([0.005] * 50)
        rs = compute_relative_strength(stock, bench, window=20)
        assert rs is not None
        assert rs < 0

    def test_none_on_insufficient_data(self):
        short = pd.Series([100.0, 101.0])
        bench = pd.Series([100.0, 100.5])
        assert compute_relative_strength(short, bench, window=20) is None

    def test_none_on_none_input(self):
        assert compute_relative_strength(None, None, window=20) is None
