# ETF & Stock Momentum Workbench v8

A learning-friendly research tool for tracking momentum in ETFs and stocks, understanding **why** they behave as they do over
months and years, and connecting that behaviour to the economic cycle – using **only free, open data**
(Yahoo Finance, FRED, Wikipedia index lists, Ken French's factor library). No accounts, no API keys.

> Educational research software, not investment advice. Free data can be wrong or late. Verify anything important before acting.

## Start (Windows)
1. Install Python 3.10–3.12 from python.org (tick **Add to PATH**).
2. Unzip this folder, double-click **`run_windows.bat`**. First run installs packages (a few minutes) and opens your browser.
   Mac/Linux: `./run_mac_linux.sh`.

## First 10 minutes
1. **Data Center → ① Prices**: leave *Detect trading currencies* on → **Update ETF prices** (first run takes a few minutes).
2. **Data Center → ② Economy** → **Update economic data**; optionally **③ Style factors**.
3. **Data Center → ④ Connection test** and **⑥ Self-test** – confirm every source is reachable and the engine passes its checks.
4. **ETF Screener** → today's ranking and model portfolio. **Backtest Lab** → *Run backtest*; read the *Is it luck?* verdict first.
5. **Learn** whenever a term is unfamiliar. Every setting has an ⓘ hover explanation.

## Pages
| Page | What it answers |
|---|---|
| Home / Learn | Orientation, tutorials, glossary, indicator guide, pitfalls, FAQ |
| ETF Screener | Which ETFs are strong now, and a diversified, risk-sized model portfolio with a rule-by-rule decision audit |
| Backtest Lab | How would these rules have done, with costs and delays – and could it be luck (bootstrap, Deflated Sharpe)? |
| Market Map | Where was performance strong in any month; which holdings explain the strategy's result |
| Group Explorer | Rank the members of an index/sector/custom list; breadth, concentration, who drove returns; single-stock "why" (market, style factors, own risk) |
| Asset Lab | Long-run profile, drawdowns, seasonality, trend behaviour, behaviour by economic regime, indicator lead/lag |
| Macro & Cycles | Growth / inflation / stress / yield-curve / policy regimes from FRED, known-at-the-time only |
| Compare with Economy | Which assets and US sectors move with an economic measure – ranked with reliability verdicts, lag map and rolling stability |
| Probabilities | Down / Sideways / Up odds from three views, scored out-of-sample against a naive baseline |
| Data Center / Audit & Export | Downloads, data quality, self-test, Excel report with parameters and data fingerprint |

## Know the limits
* **Survivorship bias**: free index lists are *current* members only; long histories look better than reality.
* **Macro data are revised**: FRED serves final numbers; the tool applies publication delays but cannot remove revision look-ahead.
* **Factors are US/USD**; for non-US stocks they are a rough lens.
* **Probabilities** are only as good as their out-of-sample record, shown beside every forecast. Often the honest answer is "no edge over the base rate".
* Trading currency of London lines is detected from Yahoo and cached. Override any ETF in `universe/currency_overrides.csv`.

## Your own universe / lists
Replace or add an Excel file in `universe/` (sheet `ETF_Master`, columns `Yahoo_Ticker`, `Fund_Name`, …) and type its name in the sidebar.
Stock groups: pick an index, the sample group, or paste tickers (`AAPL`, `VOD.L`, `SAP.DE`; or `TICKER,Name,Group`).

## Folders
`app.py` entry · `workbench/` engine and pages · `universe/` your lists · `data/` downloaded caches · `logs/workbench.log` · `tests/`

## Tests (optional)
`pip install pytest scikit-learn` then `pytest tests -q`. See `VALIDATION.md` for what has and has not been verified.
