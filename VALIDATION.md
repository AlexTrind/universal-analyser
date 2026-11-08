# Validation report – Workbench v8.2.3

## 1. What was wrong in v7.8 (confirmed by test, then fixed)
| Finding | Effect | v8 |
|---|---|---|
| Transaction cost applied to one leg only | Costs understated about 2× | Charged on total traded value (both legs) against drifted weights |
| Incremental cache merged old and new prices without rescaling | Artificial jumps when dividends re-adjust history | `merge_adjusted` rescales on overlap; if shapes disagree it forces a full re-download |
| Hysteresis (entry/exit ranks) defeated in construction | Excess turnover | Built into portfolio construction |
| Missing price on a holding silently dropped the leg | Wrong returns | Flags table; holding is carried, never silently removed |
| Currency guessed from ticker suffix | USD-quoted London lines mis-converted | Workbook override → cached Yahoo currency → Yahoo lookup → suffix guess (flagged as a guess); pence handled |
| Features recomputed per date in loops | Slow | Vectorised point-in-time engine; **23–26× faster** (measured on synthetic data in the build environment) |

## 2. Tests (97 automated, all passing at build time)
* **Parity**: new engine reproduces v7.8 rankings/scores on the same data (same definitions, so results stay comparable).
* **Hand-calculation checks**: drawdown, 12-1 momentum, Newey-West errors (vs explicit loop), Wilson interval, logistic-model optimum (gradient ≈ 1e-7; cross-checked against scikit-learn in tests).
* **Accounting**: backtest contributions sum to gross return; compounded periods equal the daily curve; cost = traded × bps.
* **No look-ahead / leakage**: scrambling all future prices leaves earlier backtest decisions unchanged; forecasts at month *t* never use data after *t*; macro values are unknown until their publication lag has passed.
* **Statistics**: false-positive rates of Deflated Sharpe, circular-shift permutation test, bootstrap and the "no edge" forecast verdict are controlled on pure noise; planted effects (momentum, regime, predictive signal, factor betas, seasonality) are detected.
* **UI smoke test** (`tests/test_ui_smoke.py`): every page's `render()` is executed against synthetic data with a fake Streamlit/Plotly that enforces unique widget keys, numeric-type consistency, defaults inside ranges and a hover-help text on every input. Also covers empty-data behaviour, the stock source, and every "Update…" button with the network replaced by synthetic data. This found and fixed duplicate widget keys and a time-zone bug.
* In-app **Self-test** (Data Center ⑥) re-runs 11 core checks on your machine.

## 3. NOT verified here – please check
1. **Live data**: this build environment had no internet. Yahoo/FRED/Wikipedia/Ken French connectivity and parsing were tested on saved/synthetic formats only. Run *Data Center → Connection test* first. Wikipedia table layouts and yfinance behaviour change occasionally.
2. **Real Streamlit/Plotly rendering**: not installed here, so appearance, chart property validity and the Streamlit version-specific behaviour are untested. If a page shows an error box, the technical details expander explains it; `logs/workbench.log` has the trace.
3. **FRED series IDs** were written from knowledge, not looked up live. A discontinued ID just drops out (shown in Data Center ②); check *Data freshness* on Macro & Cycles.
4. **Methodological limits**: survivorship bias in index lists; revised macro data; US-based style factors; macro "regimes" are simplified descriptions; probabilities have few independent observations at multi-month horizons (shown as "Independent obs").
5. Dividends/taxes/FX costs/spreads beyond the bps setting are not modelled. UK tax treatment is outside scope.

## 4. Suggested personal checks
Compare 3–4 ETF prices and 12-month returns against your broker; confirm currencies in Data Center ⑤; compare one regime label with the NBER recession shading; run the backtest with cost 25 bps and 2-day delay and see whether the verdict survives.

## 5. v8.0.1 changes
* Data check widened: any one-day price move above +150% / below -60%, or a ×/÷ 85-115 step, now FAILs the ETF (excluded from signals and comparators). Previously only an almost exact ×100 was caught, so a pence/pound-type error could slip into the "Equal-weight universe" line and make it jump.
* Backtest Lab warns when any line moves >12% in a day and names suspect tickers; explains the equal-weight line.
* Probabilities: minimum training months can go down to 24; the history needed is training + horizon + 24 (was +36); the message says what to lower.
* Help text: 0% benchmark = momentum sleeve 100% and "Unused weight goes to" = Cash.

## 6. v8.1.0 – Compare with Economy
* New page: asset-vs-indicator overlay, ranking with reliability verdicts, lead/lag heat-map (only false-discovery-surviving cells coloured), rolling stability, data-reliability table. US sector ETFs + SPY (25+ years of free history) are added alongside your assets by default.
* **Statistics validated on synthetic worlds** (tests/test_compare.py): with slow-moving noise indicators ("rho" 0.0 and 0.95) fewer than 4% of pure-noise assets are labelled Supported (about 2% observed); a planted link is recovered in 10/10 worlds at moderate strength and placed at the right lags; assets with short history are excluded, not reported.
* **Why not an ordinary t-test:** with slow-moving indicators, standard (even robust) t-tests over-reject. p-values now come from sliding the two series past each other (keeps persistence and overlapping windows, destroys real links), then false-discovery (Benjamini–Hochberg) correction across assets/cells. The Asset Lab lead/lag p-values use the same method. "Supported" additionally requires the same direction in both halves of the history and at neighbouring lags.
* Limits: the correction covers the assets/lags shown, not every indicator you browse; sector ETFs are USD; macro series are revised data.

## 7. v8.2.0 – checked against the user's real downloaded data (Nov 2026)
Real files tested: 144 ETF/FX price series (2014-11 → 2026-11), 42 stock/sector series, 29 FRED series, `price_meta.json`, `workbench.log`.

| Finding in the real data | Effect | Fix |
|---|---|---|
| **SMEA.L** quoted ~£60 until 24 Nov 2023, then ~6,885p | the "equal-weight universe" line jumped +85 % in one day (final value 2.92 instead of 1.57) – this was the sharp rise | automatic ×100 rescale of the older history (`repair_unit_errors`), reported as `UNIT_REPAIRED`; if the step is not a clean ×100 the series still FAILs and is excluded |
| **INDU.L** (1 day) and **WOOD.L** (≈5 days) quoted ÷100 | fake −99 % / +9,800 % days | same repair |
| **COFF.L, COPA.L, NGAS.L, XSTR.L, WQDV.L, IUMO.L, IBTU.L** years of stale/flat prices | returns before the live period are meaningless | new `FLAT_PERIOD` warning |
| Macro table contained an unfinished month (6 days of November) | a half-month VIX average counted as a full month | months that have not finished are no longer published |
| Backtest page crashed (`NoneType − NoneType`) when the period had too little history | red error screen | friendly message instead |

After the repair: equal-weight 2021-2026 = 1.57× (CAGR 8.1 %), benchmark 1.92×, strategy 1.87×.
Checked and working on real data: the 29 FRED files load (no missing or empty series), regimes classify, Compare-with-Economy ranks the 11 sector ETFs + S&P 500 against T10Y3M / UNRATE / VIX / oil, and the Probabilities forecast runs for ^GSPC (145 months) and VWRP.L (89 months, needs "Min training months" ≤ ~40).
Still NOT checked here: the Yahoo/FRED connections themselves, and how pages look in a browser.
Tests: 114 automated (incl. v7.8 parity), all passing.

## 8. v8.2.1 – Compare with Economy: stock selection and sector co-plot
* Reported: choosing a stock did not update the overlay graph, and no sector ETF was shown with it.
* Cause: the overlay used a separate picker that kept its old choice, and the app had no stock→sector link.
* Fix: the overlay now starts with every chosen asset plus each stock's own sector ETF and follows the selection; sectors come from the index lists already downloaded in Group Explorer, a saved Yahoo lookup, or your own choice (all remembered in `data/stocks/sector_map.json`).
* Tests: 117 automated, all passing. The live Yahoo sector lookup itself could not be exercised without network access.

## 9. v8.2.2 – chart range and history length
* Overlay lines were cut to the overlap of ALL selected series (one young ETF shortened everything) – each line now uses its own history.
* Charts now always set their x-axis to the data shown, clip recession bands to it, and are re-created when the selection changes.
* Sector ETFs first saved with 12 years of history were never extended when 25 years were requested – fixed (a one-off longer download per ticker, remembered in `price_meta.json`); the page offers an "Extend sector ETFs to 25 years" button when it detects short history.
* Tests: 118 automated, all passing.

## 10. v8.2.3
* Overlay picker kept its old selection when a stock's sector ETF was matched or changed afterwards; the picker now refreshes when the match changes. Tests: 119.

## 11. v8.2.4 – proactive audit on real data
Ran every page (backtest 27 variants, Asset Lab for every ETF and stock, Probabilities, Compare for every indicator, Macro, Screener, Map, Group, Data, Audit) against the user's real downloaded data with a probe harness (`tests/audit_harness.py`) that flags crashes, error messages, empty charts, truncated lines and bad KPIs.
- Found and fixed: Asset Lab crashed for an ETF that never rises above its 200-day average (CANA.L); regression test added.
- Compare "Lead heatmap" now says so explicitly when nothing survives the false-discovery correction instead of showing a blank grid.
- Checked and not bugs: CNX1.L (pence-quoted, £1,170 is right); currency labels cross-checked by return correlation against a reference fund (3 weak signals, none conclusive); macro "overlap" flags are composites with different start dates, each drawn on its own history.
- Not verifiable offline: live Yahoo/FRED/Wikipedia fetches, browser rendering, style-factor features.
