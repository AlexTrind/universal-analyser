import numpy as np
import pandas as pd
import pytest
from workbench import data_prices as D, quality as Q


def series(n=300, seed=1, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, .006, n))), index=idx)


def test_merge_appended_exact():
    s = series()
    m, st = D.merge_adjusted(s.iloc[:250], s.iloc[230:])
    assert st == "APPENDED" and m.index.is_unique
    assert np.allclose(m.values, s.values)


def test_merge_rescaled_after_dividend_adjustment():
    s = series()
    old = s.iloc[:250]
    new = s.iloc[230:] * 0.97 # fresh download is rescaled history (a dividend was paid)
    m, st = D.merge_adjusted(old, new)
    assert st == "RESCALED"
    assert np.allclose(m.values, s.values * 0.97)          # old history brought to new scale
    r = m.pct_change().dropna()
    assert np.allclose(r.values, s.pct_change().dropna().values)  # NO fake seam return


def test_merge_inconsistent_detected():
    s = series()
    new = s.iloc[230:].copy()
    new.iloc[5:] *= 1.05 # shape differs inside overlap
    _, st = D.merge_adjusted(s.iloc[:250], new)
    assert st == "INCONSISTENT"


def test_merge_no_overlap_flagged():
    s = series()
    _, st = D.merge_adjusted(s.iloc[:100], s.iloc[150:])
    assert st == "NO_OVERLAP"


def test_flatten_both_level_orders_and_single():
    idx = pd.bdate_range("2024-01-01", periods=3)
    a = pd.DataFrame(np.arange(6.).reshape(3, 2) + 1, index=idx, columns=["A", "B"])
    m1 = pd.concat({"Close": a, "Volume": a * 0}, axis=1)           # (Price, Ticker)
    m2 = m1.swaplevel(axis=1)                                         # (Ticker, Price)
    for m in (m1, m2):
        out = D.flatten_download(m, ["A", "B"])
        assert list(out.columns) == ["A", "B"] and np.allclose(out.values, a.values)
    single = pd.DataFrame({"Close": [1., 2, 3], "Open": [1, 1, 1]}, index=idx)
    assert list(D.flatten_download(single, ["Z"]).columns) == ["Z"]
    assert D.flatten_download(pd.DataFrame(), ["A"]).empty


class FakeFeed:
    """Pretends to be Yahoo: serves slices of known series, can fail on demand."""
    def __init__(self, data):
        self.data, self.calls, self.fail = data, [], set()
    def __call__(self, tickers, start, end):
        self.calls.append((tuple(tickers), start))
        cols = {t: self.data[t] for t in tickers if t in self.data and t not in self.fail}
        if not cols:
            return pd.DataFrame()
        df = pd.DataFrame(cols)
        return df.loc[start:end]


def test_store_full_then_incremental_with_dividend_rescale(tmp_path):
    full = {"AAA.L": series(400, 1), "BBB.L": series(400, 2)}
    feed = FakeFeed({k: v.iloc[:300] for k, v in full.items()})
    store = D.PriceStore(tmp_path)
    today1 = full["AAA.L"].index[299] + pd.Timedelta(days=1)
    r1 = store.update(["AAA.L", "BBB.L"], fetcher=feed, today=today1)
    assert set(r1.status.Source) == {"FRESH_FULL"}
    # later: provider data now includes 100 more days and AAA history was rescaled by a dividend
    feed.data = {"AAA.L": full["AAA.L"] * 0.98, "BBB.L": full["BBB.L"]}
    today2 = full["AAA.L"].index[399] + pd.Timedelta(days=1)
    r2 = store.update(["AAA.L", "BBB.L"], fetcher=feed, today=today2)
    st = r2.status.set_index("Ticker")
    assert st.at["AAA.L", "Detail"] == "RESCALED" and st.at["BBB.L", "Detail"] == "APPENDED"
    assert st.at["AAA.L", "Source"] == "FRESH_INCREMENTAL"
    # the incremental call must not have re-downloaded years of history
    assert feed.calls[-1][1] > "2025-01-01"
    got = r2.prices["AAA.L"].dropna()
    assert np.allclose(got.pct_change().dropna().values, full["AAA.L"].pct_change().dropna().values)
    assert store.load().shape == r2.prices.shape # persisted


def test_store_failed_download_keeps_cache_and_says_so(tmp_path):
    s = series(300, 3)
    feed = FakeFeed({"AAA.L": s})
    store = D.PriceStore(tmp_path)
    store.update(["AAA.L"], fetcher=feed, today=s.index[-1] + pd.Timedelta(days=1))
    feed.fail = {"AAA.L"}
    r = store.update(["AAA.L", "ZZZ.L"], fetcher=feed, today=s.index[-1] + pd.Timedelta(days=5))
    st = r.status.set_index("Ticker").Source
    assert st["AAA.L"] == "STALE_CACHE" and st["ZZZ.L"] == "UNRESOLVED"
    assert len(r.prices["AAA.L"].dropna()) == 300 and r.errors


def test_bad_tick_removed_but_real_crash_kept():
    s = series(100, 4)
    s2 = s.copy(); s2.iloc[50] *= 1.6                     # spike that reverses
    clean, bad = Q.repair_bad_ticks(s2)
    assert bad == [s.index[50]] and np.isnan(clean.iloc[50])
    s3 = s.copy(); s3.iloc[50:] *= 0.6                    # genuine crash that stays
    _, bad3 = Q.repair_bad_ticks(s3)
    assert bad3 == []


def test_currency_priority_and_pence():
    ct = D.resolve_currencies(["CSPX.L", "VWRP.L", "XYZ.L", "EXS1.DE", "AAPL"],
                              overrides={"CSPX.L": "USD"}, cached={"VWRP.L": "GBp"},
                              lookup=lambda t: None)
    d = ct.set_index("Ticker")
    assert d.at["CSPX.L", "Currency"] == "USD" and d.at["CSPX.L", "CurrencySource"] == "WORKBOOK"
    assert d.at["VWRP.L", "Currency"] == "GBP" and d.at["VWRP.L", "Scale"] == 0.01
    assert d.at["XYZ.L", "CurrencySource"] == "GUESSED_FROM_SUFFIX"
    assert d.at["EXS1.DE", "Currency"] == "EUR" and d.at["AAPL", "Currency"] == "USD"


def test_fx_conversion_numbers():
    idx = pd.bdate_range("2024-01-01", periods=5)
    native = pd.DataFrame({"US1": [100., 101, 102, 103, 104], "USDGBP=X": [.8, .8, .79, .79, np.nan]}, index=idx)
    ct = D.resolve_currencies(["US1"], overrides={"US1": "USD"})
    p, st = D.to_base_currency(native, ct, "GBP")
    assert p["US1"].iloc[0] == pytest.approx(80.0) and p["US1"].iloc[2] == pytest.approx(102 * .79)
    assert p["US1"].iloc[4] == pytest.approx(104 * .79)       # FX carried forward, not dropped
    assert st.Comparable.all()
    p2, st2 = D.to_base_currency(native.drop(columns="USDGBP=X"), ct, "GBP")
    assert "US1" not in p2 and not st2.Comparable.any() and "FX_MISSING" in st2.Status[0]
    # pence: 12345 GBp -> 123.45 GBP
    nat = pd.DataFrame({"P": [12345., 12400]}, index=idx[:2])
    p3, _ = D.to_base_currency(nat, D.resolve_currencies(["P"], overrides={"P": "GBp"}), "GBP")
    assert p3["P"].iloc[0] == pytest.approx(123.45)


def test_quality_flags():
    ref = pd.Timestamp("2026-11-06")
    idx = pd.bdate_range(end=ref, periods=400)
    ok = pd.Series(100.0, index=idx) * np.exp(np.cumsum(np.random.default_rng(1).normal(0, .005, 400)))
    unit = ok.copy(); unit.iloc[200:] *= 100
    stale = ok.iloc[:-30]
    df = pd.concat({"OK": ok, "UNIT": unit, "STALE": stale}, axis=1)
    rep = Q.quality_report(df, ref).set_index("Ticker")
    assert rep.at["OK", "Severity"] == "OK"
    assert rep.at["UNIT", "Severity"] == "FAIL" and "UNIT_SWITCH" in rep.at["UNIT", "Issues"]
    assert rep.at["STALE", "Severity"] == "FAIL"
    assert Q.failing_tickers(rep.reset_index()) == {"UNIT", "STALE"}
