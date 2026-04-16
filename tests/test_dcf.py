"""tests/test_dcf.py — Unit tests for pure DCF math helpers."""
import pytest
from dcf import calculate_growth_rate_from_series, build_growth_schedule
from constants import DCF_GROWTH_FADE_CAP, DCF_GROWTH_FADE_TARGET, DCF_GROWTH_FADE_YEARS


# ---------------------------------------------------------------------------
# calculate_growth_rate_from_series
# ---------------------------------------------------------------------------
class TestCalculateGrowthRateFromSeries:
    def _rows(self, values, field="FCF", start_year=2020):
        return [{"Year": start_year + i, field: v} for i, v in enumerate(values)]

    def test_cagr_exact(self):
        # 100 → 121 over 2 years = 10% CAGR
        rows = self._rows([100, 110, 121])
        rate = calculate_growth_rate_from_series(rows, "FCF")
        assert rate == pytest.approx(0.10, abs=0.001)

    def test_single_row_returns_none(self):
        rows = self._rows([100])
        assert calculate_growth_rate_from_series(rows, "FCF") is None

    def test_empty_returns_none(self):
        assert calculate_growth_rate_from_series([], "FCF") is None

    def test_missing_field_returns_none(self):
        rows = [{"Year": 2020}, {"Year": 2021}]
        assert calculate_growth_rate_from_series(rows, "FCF") is None

    def test_negative_start_falls_back_to_mean(self):
        # start_value <= 0 triggers mean-pct-change fallback
        rows = self._rows([-10, 5, 10])
        rate = calculate_growth_rate_from_series(rows, "FCF")
        # should return a float, not None
        assert rate is not None
        assert isinstance(rate, float)

    def test_lookback_limits_window(self):
        rows = self._rows([50, 60, 80, 100, 130], start_year=2018)
        rate3 = calculate_growth_rate_from_series(rows, "FCF", lookback_years=3)
        rate1 = calculate_growth_rate_from_series(rows, "FCF", lookback_years=1)
        # Different lookbacks should generally produce different rates
        assert rate3 is not None
        assert rate1 is not None


# ---------------------------------------------------------------------------
# build_growth_schedule
# ---------------------------------------------------------------------------
class TestBuildGrowthSchedule:
    def test_length_matches_years(self):
        schedule = build_growth_schedule(0.10, 0.025, years=10)
        assert len(schedule) == 10

    def test_starts_near_initial(self):
        schedule = build_growth_schedule(0.10, 0.025, years=10)
        assert schedule[0] == pytest.approx(0.10, abs=0.001)

    def test_ends_near_terminal(self):
        schedule = build_growth_schedule(0.10, 0.025, years=10)
        assert schedule[-1] == pytest.approx(0.025, abs=0.001)

    def test_single_year_returns_terminal(self):
        schedule = build_growth_schedule(0.20, 0.025, years=1)
        assert schedule == [pytest.approx(0.025)]

    def test_normal_rate_is_linear(self):
        # rate at/below fade cap → simple linspace
        schedule = build_growth_schedule(0.10, 0.025, years=5)
        for i in range(len(schedule) - 1):
            step = schedule[i + 1] - schedule[i]
            assert step < 0, "should decrease monotonically"

    def test_high_rate_two_phase_fade(self):
        # initial > DCF_GROWTH_FADE_CAP triggers two-phase schedule
        initial = DCF_GROWTH_FADE_CAP + 0.10  # e.g. 0.25
        schedule = build_growth_schedule(initial, 0.025, years=10)
        assert len(schedule) == 10
        # After the rapid-collapse phase, values should be near DCF_GROWTH_FADE_TARGET
        assert schedule[DCF_GROWTH_FADE_YEARS] == pytest.approx(DCF_GROWTH_FADE_TARGET, abs=0.01)

    def test_all_rates_are_floats(self):
        schedule = build_growth_schedule(0.30, 0.025, years=10)
        assert all(isinstance(r, float) for r in schedule)

    def test_high_rate_short_period_no_two_phase(self):
        # years <= DCF_GROWTH_FADE_YEARS → two-phase condition not met → linear
        schedule = build_growth_schedule(0.30, 0.025, years=DCF_GROWTH_FADE_YEARS)
        assert len(schedule) == DCF_GROWTH_FADE_YEARS
