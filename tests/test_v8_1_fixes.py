import numpy as np
import pandas as pd

from workbench import forecast as FC, quality as Q


def _px(n=600, seed=0):
    r = np.random.default_rng(seed).normal(0.0003, 0.01, n)
    return pd.Series(50 * np.exp(np.cumsum(r)), index=pd.bdate_range("2022-01-03", periods=n))


def test_permanent_step_of_any_size_is_a_fail():
    for factor in (40.0, 96.0, 100.0, 104.7, 0.01, 0.3):
        s = _px()
        s.iloc[300:] *= factor
        sev = Q.quality_report(s.to_frame("X")).iloc[0]
        assert sev["Severity"] == "FAIL", (factor, sev["Issues"])


def test_normal_series_and_real_big_moves_not_failed():
    s = _px()
    s.iloc[300:] *= 0.8          # a 20% one-day crash is possible; WARN at most, never FAIL
    assert Q.quality_report(s.to_frame("X")).iloc[0]["Severity"] in ("OK", "INFO")


def test_forecast_message_tells_user_what_to_change():
    import re
    me = _px(110 * 21).resample("ME").last().iloc[:110]
    res = FC.run_forecast(me, None, None, h=3, use_price=True, use_macro=False, min_train=78)
    assert not res.ok and "Lower" in res.reason, res.reason
    k = int(re.search(r"to (\d+) or less", res.reason).group(1))
    res2 = FC.run_forecast(me, None, None, h=3, use_price=True, use_macro=False, min_train=k)
    assert res2.ok, res2.reason
