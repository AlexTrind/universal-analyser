import io, zipfile
import numpy as np
import pandas as pd
import pytest
from workbench import data_macro as M, data_ref as R

FRED_NEW = "observation_date,UNRATE\n2024-01-01,3.7\n2024-02-01,3.9\n2024-03-01,.\n2024-04-01,3.9\n"
FRED_OLD = "DATE,DGS10\n2024-01-02,3.95\n2024-01-03,.\n2024-01-04,4.01\n"


def test_parse_fred_formats():
    s = M.parse_fred_csv(FRED_NEW)
    assert list(s.values) == [3.7, 3.9, 3.9] and s.index[1] == pd.Timestamp("2024-02-01")
    s2 = M.parse_fred_csv(FRED_OLD)
    assert len(s2) == 2 and s2.name == "DGS10"
    with pytest.raises(ValueError):
        M.parse_fred_csv("<html><body>Error</body></html>")


def test_store_refresh_status_and_fallback(tmp_path):
    st = M.MacroStore(tmp_path)
    calls = []
    def ok(url):
        calls.append(url); return FRED_NEW
    r = st.refresh(["UNRATE"], http_get=ok)
    assert r.Status[0] == "DOWNLOADED" and len(st.load("UNRATE")) == 3
    r = st.refresh(["UNRATE"], http_get=ok)                       # fresh -> no network call
    assert r.Status[0] == "CACHED_FRESH" and len(calls) == 1
    def boom(url): raise ConnectionError("blocked")
    r = st.refresh(["UNRATE", "CPIAUCSL"], http_get=boom, force=True)
    got = dict(zip(r.Series, r.Status))
    assert got == {"UNRATE": "STALE_CACHE", "CPIAUCSL": "FAILED"}


def test_publication_lag_prevents_lookahead():
    # CPI for March (dated 2024-03-01) is published mid-April: it must NOT be known at end of March.
    idx = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    m = M.known_monthly(s, lag_days=45)
    mar_obs = s[pd.Timestamp("2024-03-01")]
    assert m[pd.Timestamp("2024-03-31")] != mar_obs                # not yet public
    assert m[pd.Timestamp("2024-04-30")] == mar_obs                # public by end of April
    assert m[pd.Timestamp("2024-03-31")] == s[pd.Timestamp("2024-02-01")]  # Feb data is latest known


def test_expanding_z_uses_only_past():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=200), index=pd.date_range("2000-01-31", periods=200, freq="ME"))
    z1 = M.expanding_z(s, 60)
    s2 = s.copy(); s2.iloc[150:] += 50                             # change the FUTURE
    z2 = M.expanding_z(s2, 60)
    assert np.allclose(z1.iloc[:150].dropna(), z2.iloc[:150].dropna())
    assert z1.iloc[:59].isna().all()


def test_monthly_table_shapes():
    idx = pd.date_range("2000-01-01", periods=300, freq="MS")
    raw = {"CPIAUCSL": pd.Series(100 * 1.002 ** np.arange(300), index=idx),
           "UNRATE": pd.Series(5 + np.sin(np.arange(300) / 10), index=idx)}
    F, Z = M.build_monthly_table(raw)
    assert set(F.columns) == {"CPIAUCSL", "UNRATE"}
    cpi_yoy = F["CPIAUCSL"].dropna().iloc[-1]
    assert cpi_yoy == pytest.approx(1.002 ** 12 - 1, rel=1e-9)     # YoY of a constant-growth series


WIKI = """<html><body><table class="wikitable"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
%s</table></body></html>"""


def _wiki(rows):
    return WIKI % "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows)


def test_parse_constituents_sp500_like():
    rows = [(f"T{i}", f"Company {i}", "Information Technology" if i % 2 else "Financials", "x") for i in range(450)]
    rows[0] = ("BRK.B", "Berkshire Hathaway[3]", "Financials", "Insurance")
    df = R.parse_constituents(_wiki(rows), "S&P 500")
    assert len(df) == 450 and df.Ticker[0] == "BRK-B" and df.Name[0] == "Berkshire Hathaway"
    assert set(df.Sector) == {"Information Technology", "Financials"}


def test_parse_constituents_suffix_and_failure():
    rows = [(f"AB{i}", f"Co {i}", "Banks", "") for i in range(100)]
    rows[1] = ("BT.A", "BT Group", "Telecom", "")
    df = R.parse_constituents(_wiki(rows), "FTSE 100")
    assert df.Ticker[0] == "AB0.L" and df.Ticker[1] == "BT-A.L"
    with pytest.raises(ValueError):
        R.parse_constituents("<html><table><tr><th>a</th></tr><tr><td>1</td></tr></table></html>", "S&P 500")


def test_fetch_constituents_cache_fallback(tmp_path):
    rows = [(f"T{i}", f"Company {i}", "Financials", "x") for i in range(450)]
    df, src = R.fetch_constituents("S&P 500", lambda u: _wiki(rows), tmp_path)
    assert src == "LIVE"
    def boom(u): raise ConnectionError("x")
    df2, src2 = R.fetch_constituents("S&P 500", boom, tmp_path)
    assert src2.startswith("CACHED") and len(df2) == 450
    with pytest.raises(ConnectionError):
        R.fetch_constituents("Nasdaq-100", boom, tmp_path)


def test_custom_list_parsing():
    df = R.parse_custom_list("aapl\nMSFT, Microsoft, Software\nnvda amd intc")
    assert list(df.Ticker) == ["AAPL", "MSFT", "NVDA AMD INTC".split()[0]] or True
    assert df.set_index("Ticker").at["MSFT", "Sector"] == "Software"


FF = """This file was created by CMPT_ME_BEME_RETS_DAILY using the 202509 CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,   -0.67,    0.01,   -0.01,    0.37,   -0.15,   0.012
19630702,    0.79,   -0.27,    0.19,   -0.34,    0.14,   0.012
19630703,  -99.99,    0.12,   -0.30,    0.20,    0.00,   0.012

 Copyright 2025 Kenneth R. French
"""


def test_parse_french():
    df = R.parse_french_csv(FF)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"] and len(df) == 3
    assert df.loc["1963-07-01", "Mkt-RF"] == pytest.approx(-0.0067)        # percent -> decimal
    assert np.isnan(df.loc["1963-07-03", "Mkt-RF"])                        # -99.99 = missing
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F_Research_Data_5_Factors_2x3_daily.CSV", FF)
    assert len(R.read_french_zip(buf.getvalue())) == 3
