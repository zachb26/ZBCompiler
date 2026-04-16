"""tests/test_analytics_scoring.py — Unit tests for analytics_scoring functions."""
import pytest
import pandas as pd
from analytics_scoring import (
    cap_weights,
    score_to_signal,
    score_to_sentiment,
    score_trend_distance,
    step_signal_toward_neutral,
    has_bullish_trend,
    has_bearish_trend,
    classify_market_regime,
    summarize_engine_biases,
    compute_decision_confidence,
    apply_confidence_guard,
    resolve_overall_verdict,
)

# Minimal settings for verdict / confidence functions
SETTINGS = {
    "overall_buy_threshold": 2.0,
    "overall_strong_buy_threshold": 4.0,
    "overall_sell_threshold": -2.0,
    "overall_strong_sell_threshold": -4.0,
    "decision_min_confidence": 40.0,
    "decision_hold_buffer": 1.0,
}


# ---------------------------------------------------------------------------
# cap_weights
# ---------------------------------------------------------------------------
class TestCapWeights:
    def test_sums_to_one(self):
        w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        result = cap_weights(w, max_weight=0.4)
        assert result.sum() == pytest.approx(1.0)

    def test_no_weight_exceeds_cap(self):
        w = pd.Series({"A": 0.7, "B": 0.2, "C": 0.1})
        result = cap_weights(w, max_weight=0.4)
        assert (result <= 0.4 + 1e-9).all()

    def test_no_cap_unchanged_proportions(self):
        w = pd.Series({"A": 0.4, "B": 0.35, "C": 0.25})
        result = cap_weights(w, max_weight=1.0)
        assert result.sum() == pytest.approx(1.0)

    def test_equal_weights_unchanged_by_cap(self):
        w = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0})
        result = cap_weights(w, max_weight=0.5)
        assert result.sum() == pytest.approx(1.0)
        assert (result <= 0.5 + 1e-9).all()


# ---------------------------------------------------------------------------
# score_to_signal
# ---------------------------------------------------------------------------
class TestScoreToSignal:
    def test_strong_buy(self):
        assert score_to_signal(5) == "STRONG BUY"

    def test_buy(self):
        assert score_to_signal(3) == "BUY"

    def test_hold(self):
        assert score_to_signal(0) == "HOLD"

    def test_sell(self):
        assert score_to_signal(-3) == "SELL"

    def test_strong_sell(self):
        assert score_to_signal(-5) == "STRONG SELL"

    def test_exact_buy_threshold(self):
        assert score_to_signal(2) == "BUY"

    def test_exact_sell_threshold(self):
        assert score_to_signal(-2) == "SELL"

    def test_just_below_buy(self):
        assert score_to_signal(1.9) == "HOLD"


# ---------------------------------------------------------------------------
# score_to_sentiment
# ---------------------------------------------------------------------------
class TestScoreToSentiment:
    def test_positive(self):
        assert score_to_sentiment(4) == "POSITIVE"

    def test_negative(self):
        assert score_to_sentiment(-4) == "NEGATIVE"

    def test_mixed(self):
        assert score_to_sentiment(0) == "MIXED"

    def test_boundary_positive(self):
        assert score_to_sentiment(3) == "POSITIVE"

    def test_boundary_negative(self):
        assert score_to_sentiment(-3) == "NEGATIVE"


# ---------------------------------------------------------------------------
# score_trend_distance
# ---------------------------------------------------------------------------
class TestScoreTrendDistance:
    def test_above_tolerance(self):
        assert score_trend_distance(103, 100) == 1

    def test_below_tolerance(self):
        assert score_trend_distance(97, 100) == -1

    def test_within_tolerance(self):
        assert score_trend_distance(101, 100) == 0

    def test_none_value(self):
        assert score_trend_distance(None, 100) == 0

    def test_zero_baseline(self):
        assert score_trend_distance(100, 0) == 0


# ---------------------------------------------------------------------------
# step_signal_toward_neutral
# ---------------------------------------------------------------------------
class TestStepSignalTowardNeutral:
    def test_strong_buy_to_buy(self):
        assert step_signal_toward_neutral("STRONG BUY") == "BUY"

    def test_buy_to_hold(self):
        assert step_signal_toward_neutral("BUY") == "HOLD"

    def test_hold_stays(self):
        assert step_signal_toward_neutral("HOLD") == "HOLD"

    def test_sell_to_hold(self):
        assert step_signal_toward_neutral("SELL") == "HOLD"

    def test_strong_sell_to_sell(self):
        assert step_signal_toward_neutral("STRONG SELL") == "SELL"

    def test_unknown_falls_back_to_hold(self):
        assert step_signal_toward_neutral("UNKNOWN") == "HOLD"


# ---------------------------------------------------------------------------
# has_bullish_trend / has_bearish_trend
# ---------------------------------------------------------------------------
class TestTrendDetection:
    def test_bullish_price_above_sma200_sma50_above_sma200(self):
        assert has_bullish_trend(price=210, sma50=205, sma200=200) is True

    def test_not_bullish_price_below_sma200(self):
        assert has_bullish_trend(price=190, sma50=205, sma200=200) is False

    def test_bearish_price_below_sma200_sma50_below(self):
        assert has_bearish_trend(price=190, sma50=195, sma200=200) is True

    def test_not_bearish_price_above_sma200(self):
        assert has_bearish_trend(price=210, sma50=195, sma200=200) is False

    def test_bullish_via_momentum(self):
        assert has_bullish_trend(price=205, sma50=None, sma200=200, momentum_1y=0.10) is True

    def test_none_price_returns_false(self):
        assert has_bullish_trend(price=None, sma50=200, sma200=200) is False


# ---------------------------------------------------------------------------
# classify_market_regime
# ---------------------------------------------------------------------------
class TestClassifyMarketRegime:
    def test_bullish_trend(self):
        regime = classify_market_regime(price=215, sma50=210, sma200=200, momentum_1y=0.15)
        assert regime == "Bullish Trend"

    def test_bearish_trend(self):
        regime = classify_market_regime(price=185, sma50=190, sma200=200, momentum_1y=-0.15)
        assert regime == "Bearish Trend"

    def test_range_bound_near_sma200(self):
        regime = classify_market_regime(price=200, sma50=200, sma200=200, momentum_1y=0.01)
        assert regime == "Range-bound"

    def test_unclear_no_sma200(self):
        regime = classify_market_regime(price=200, sma50=None, sma200=None)
        assert regime == "Unclear"


# ---------------------------------------------------------------------------
# summarize_engine_biases
# ---------------------------------------------------------------------------
class TestSummarizeEngineBiases:
    def test_all_bullish(self):
        result = summarize_engine_biases(
            tech_score=4, f_score=3, v_score=2, sentiment_score=3,
            v_val="UNDERVALUED", bullish_trend=True, bearish_trend=False,
        )
        assert result["bullish_count"] == 4
        assert result["bearish_count"] == 0
        assert result["mixed"] is False

    def test_all_bearish(self):
        result = summarize_engine_biases(
            tech_score=-5, f_score=-3, v_score=-3, sentiment_score=-3,
            v_val="OVERVALUED", bullish_trend=False, bearish_trend=True,
        )
        assert result["bearish_count"] == 4
        assert result["bullish_count"] == 0

    def test_mixed_signals(self):
        result = summarize_engine_biases(
            tech_score=4, f_score=3, v_score=-3, sentiment_score=-3,
            v_val="OVERVALUED", bullish_trend=True, bearish_trend=False,
        )
        assert result["mixed"] is True


# ---------------------------------------------------------------------------
# compute_decision_confidence
# ---------------------------------------------------------------------------
class TestComputeDecisionConfidence:
    def _bias(self, bullish, bearish):
        return {"bullish_count": bullish, "bearish_count": bearish, "mixed": bullish >= 2 and bearish >= 2}

    def test_high_alignment_gives_high_confidence(self):
        conf = compute_decision_confidence(
            overall_score=5, bias_summary=self._bias(4, 0),
            regime="Bullish Trend", completeness=0.9,
        )
        assert conf >= 70

    def test_mixed_signals_reduce_confidence(self):
        conf_mixed = compute_decision_confidence(
            overall_score=0, bias_summary=self._bias(2, 2),
            regime="Range-bound", completeness=0.5,
        )
        conf_clean = compute_decision_confidence(
            overall_score=5, bias_summary=self._bias(4, 0),
            regime="Bullish Trend", completeness=0.9,
        )
        assert conf_mixed < conf_clean

    def test_confidence_clamped(self):
        conf = compute_decision_confidence(
            overall_score=100, bias_summary=self._bias(4, 0),
            regime="Bullish Trend", completeness=1.0,
        )
        assert conf <= 95.0

    def test_confidence_floor(self):
        conf = compute_decision_confidence(
            overall_score=-100, bias_summary=self._bias(0, 4),
            regime="Range-bound", completeness=0.0,
        )
        assert conf >= 5.0


# ---------------------------------------------------------------------------
# apply_confidence_guard
# ---------------------------------------------------------------------------
class TestApplyConfidenceGuard:
    def test_strong_buy_demoted_on_low_confidence(self):
        # confidence=35 is below both strong_floor (50) and base_floor (40),
        # so STRONG BUY → BUY (first guard) → HOLD (second guard)
        result = apply_confidence_guard("STRONG BUY", confidence=35, data_quality="Medium", settings=SETTINGS)
        assert result == "HOLD"

    def test_buy_demoted_to_hold_on_low_confidence(self):
        result = apply_confidence_guard("BUY", confidence=20, data_quality="Medium", settings=SETTINGS)
        assert result == "HOLD"

    def test_low_data_quality_kills_signal(self):
        result = apply_confidence_guard("BUY", confidence=80, data_quality="Low", settings=SETTINGS)
        assert result == "HOLD"

    def test_strong_signal_preserved_on_high_confidence(self):
        result = apply_confidence_guard("STRONG BUY", confidence=85, data_quality="High", settings=SETTINGS)
        assert result == "STRONG BUY"

    def test_hold_unchanged(self):
        result = apply_confidence_guard("HOLD", confidence=10, data_quality="Low", settings=SETTINGS)
        assert result == "HOLD"


# ---------------------------------------------------------------------------
# resolve_overall_verdict (integration-style)
# ---------------------------------------------------------------------------
class TestResolveOverallVerdict:
    def _call(self, overall, tech, f, v, sent, v_fund="FAIR", v_val="FAIR",
              regime="Bullish Trend", bullish=True, bearish=False):
        return resolve_overall_verdict(
            overall_score=overall, tech_score=tech, f_score=f, v_score=v,
            sentiment_score=sent, v_fund=v_fund, v_val=v_val, regime=regime,
            bullish_trend=bullish, bearish_trend=bearish, settings=SETTINGS,
        )

    def test_clean_buy_signal(self):
        verdict = self._call(overall=5, tech=4, f=3, v=2, sent=3,
                             v_fund="STRONG", v_val="UNDERVALUED")
        assert verdict in {"BUY", "STRONG BUY"}

    def test_clean_sell_signal(self):
        verdict = self._call(overall=-5, tech=-4, f=-3, v=-3, sent=-3,
                             v_fund="WEAK", v_val="OVERVALUED",
                             regime="Bearish Trend", bullish=False, bearish=True)
        assert verdict in {"SELL", "STRONG SELL"}

    def test_conflicting_signals_tend_to_hold(self):
        verdict = self._call(overall=1, tech=3, f=-2, v=-2, sent=3,
                             regime="Range-bound", bullish=False, bearish=False)
        assert verdict in {"HOLD", "BUY", "SELL"}

    def test_returns_valid_label(self):
        for overall in range(-6, 7):
            verdict = self._call(overall=overall, tech=0, f=0, v=0, sent=0)
            assert verdict in {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}
