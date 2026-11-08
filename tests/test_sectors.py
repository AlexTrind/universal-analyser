import json
import pandas as pd
from workbench import sectors as SC


def test_label_mapping_covers_gics_and_yahoo_names():
    for lab, etf in [("Information Technology", "XLK"), ("Technology", "XLK"), ("Health Care", "XLV"), ("Healthcare", "XLV"),
                     ("Consumer Cyclical", "XLY"), ("Consumer Defensive", "XLP"), ("Financial Services", "XLF"),
                     ("Basic Materials", "XLB"), ("Communication Services", "XLC"), ("Real Estate", "XLRE"), ("Utilities", "XLU"),
                     ("Energy", "XLE"), ("Industrials", "XLI"), ("Consumer Discretionary", "XLY"), ("Consumer Staples", "XLP")]:
        assert SC.etf_for_label(lab) == etf
    assert SC.etf_for_label("Unclassified") is None and SC.etf_for_label(None) is None


def test_book_uses_index_list_then_lookup_then_manual(tmp_path):
    cd = tmp_path / "c"; cd.mkdir()
    pd.DataFrame({"Ticker": ["AAPL", "XOM"], "Name": ["a", "x"], "Sector": ["Information Technology", "Energy"]}).to_csv(cd / "sp.csv", index=False)
    b = SC.SectorBook(tmp_path / "s", cd)
    assert b.etf("AAPL") == ("XLK", "index list") and b.etf("XOM")[0] == "XLE"
    assert b.etf("SHEL.L")[0] is None
    fails = b.lookup_missing(["SHEL.L", "ZZZ"], fetch=lambda t: "Energy" if t == "SHEL.L" else (_ for _ in ()).throw(RuntimeError("x")))
    assert b.etf("SHEL.L") == ("XLE", "saved lookup") and "ZZZ" in fails
    b.set_manual("AAPL", "XLC")
    assert SC.SectorBook(tmp_path / "s", cd).etf("AAPL") == ("XLC", "chosen by you")   # survives a restart
    b.set_manual("AAPL", "(none)")
    assert b.etf("AAPL")[0] is None


def test_remember_group_sectors(tmp_path):
    b = SC.SectorBook(tmp_path, tmp_path / "none")
    b.remember(pd.DataFrame({"Ticker": ["MSFT", "JPM"], "Sector": ["Information Technology", "Financials"]}))
    assert b.etf("MSFT")[0] == "XLK" and b.etf("JPM")[0] == "XLF"
