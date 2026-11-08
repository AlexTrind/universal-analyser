"""All explanatory text in one place: tooltips (HELP), glossary, tutorials, pitfalls, FAQ.

Written for an intelligent non-specialist.  Nothing here is investment advice."""
from __future__ import annotations

DISCLAIMER = (
    "Educational research software – not investment advice. Past performance does not predict future results. "
    "All numbers depend on free third-party data that can be wrong or late. Verify anything important independently "
    "before you act, and never invest money you cannot afford to lose."
)

# ----------------------------------------------------------------------------------------------
# Hover help for every control (shown by the little ⓘ / question-mark next to each widget)
# ----------------------------------------------------------------------------------------------
HELP = {
    "cmp_indicator": "The economic measure to compare against. 'Composites' blend several indicators (Growth, Inflation, Financial stress, Liquidity); the rest are single FRED series, already converted to changes or growth rates.",
    "cmp_assets": "ETFs and stocks to compare. Anything you downloaded (ETF universe or Group Explorer stocks) can be chosen.",
    "cmp_sector": "Adds the 11 US sector ETFs (technology, energy, utilities…) plus SPY alongside your choice. They have 25+ years of free history, so they give far more reliable evidence than recently launched ETFs, and show which parts of the market respond to this indicator.",
    "cmp_lag": "How many months EARLIER the indicator is observed. 0 = latest known value; 6 = the value known six months before the start of the return window. Only past values are ever used.",
    "cmp_h": "Length of the future return window in months that the indicator is compared with.",
    "cmp_minmonths": "Minimum months of overlapping data required. Assets with less are excluded rather than shown with unreliable numbers.",
    "cmp_window": "Months used for each point of the rolling correlation. Longer = smoother but slower to react.",
    "cmp_maxlag": "Largest lead (in months) to test in the lag map. More lags = more tests, so more chance flukes (this is corrected for).",
    # --- universe & benchmark
    "universe_file": "Excel workbook listing the ETFs to analyse (sheet 'ETF_Master'). Put your own file in the 'universe' folder and type its name here. New tickers need a data refresh before they appear.",
    "base_currency": "All prices are converted into this currency before comparing, so a US-dollar ETF and a pound ETF are judged on what YOU would have earned in your own money. GBP is the default for UK investors.",
    "benchmark": "The yardstick everything is compared against – normally a broad global equity fund. 'Excess return' means how much better or worse an ETF did than this benchmark. The unallocated part of your portfolio also sits here.",
    "portfolio_value": "Only used to turn percentage weights into money amounts. It does not change any signal, ranking or performance figure.",
    "active_fraction": "How much of your total portfolio is allocated to the momentum picks (the 'active sleeve'). The rest stays in the benchmark. 20% means: 20% tactical, 80% plain global index. Set it to 100% (and ‘Unused weight goes to’ = Cash) for 0% in the benchmark fund.",
    "n_active": "Maximum number of momentum ETFs to hold at once. Fewer = more concentrated and more volatile. If fewer than this many ETFs pass the rules, the extra slots stay in the benchmark – the model never forces in weak ETFs.",
    "entry_pct": "A new ETF must rank at or above this momentum percentile (100 = the strongest of the universe) AND pass the trend and positive-excess tests to be bought. Higher = pickier.",
    "exit_pct": "An ETF you already hold is only sold when its rank falls below this level (or it breaks its trend). Keeping this below the entry level avoids constant buying and selling ('hysteresis'), which saves trading costs.",
    "ma_days": "Length of the long-term moving average used as a trend filter. An ETF below its own average is considered in a downtrend and is not bought. 200 days (~10 months) is the classic choice.",
    "min_history": "ETFs with fewer trading days of price history than this are ignored. Momentum measured on a few months of data is unreliable. 252 days ≈ 1 year.",
    "sizing": "How money is split among the chosen ETFs. 'Realized volatility' gives calmer ETFs more weight (risk-based). 'Equal weight' gives each the same.",
    "corr_days": "How many past trading days are used to measure how similarly ETFs move together.",
    "corr_threshold": "Two ETFs whose returns are at least this correlated (1.0 = identical movement) are treated as the same bet and grouped into a 'cluster'. The model tries to pick from different clusters first.",
    "max_cluster": "Largest share of the momentum sleeve allowed in one cluster of look-alike ETFs. Prevents putting all your eggs in one theme.",
    "max_single": "Largest share of the momentum sleeve allowed in one ETF.",
    "cost_bps": "Cost of each trade as basis points of the amount traded (1 bp = 0.01%). 10 bps = 0.10%. It is charged on every buy AND every sell, so replacing one ETF with another costs twice this. Include spreads, fees and a little slippage.",
    "lag_days": "Delay between the signal and the trade. 1 = you decide using today's closing prices and trade at tomorrow's close – the realistic choice. 0 assumes you trade at the very price that created the signal, which flatters results.",
    "residual_to": "Where weight goes when the rules leave part of the portfolio unused: into the benchmark fund (default) or into cash earning 0%. Choose Cash if you want NO benchmark holding: the benchmark then stays only as a comparison line, not something you own.",
    "held_now": "Tick ETFs you already own. They are allowed to stay in the portfolio until their rank drops below the EXIT level (instead of needing to re-qualify for entry), which avoids needless trading.",
    "blocked": "ETFs your broker does not offer. They are removed from the actionable portfolio (their slot goes to the next-best ETF) but stay in the research ranking.",
    "asof": "Date of the analysis. Everything is calculated using only prices available up to and including this date.",
    # --- backtest
    "bt_start": "First date of the simulated history.",
    "bt_end": "Last date of the simulated history.",
    "bt_freq": "How often the model re-checks the rankings and trades. Monthly is a sensible balance; weekly trades much more and usually costs more.",
    "grid_ma": "Moving-average lengths to try in the robustness test. A real effect should work across nearby settings, not just at one magic number.",
    "grid_entry": "Entry thresholds to try in the robustness test.",
    "grid_sizing": "Sizing methods to try in the robustness test.",
    # --- groups / stocks
    "group_source": "Which set of stocks to analyse. Index lists are read live from Wikipedia; 'Custom list' lets you paste any tickers (Yahoo Finance symbols, e.g. AAPL, VOD.L, SAP.DE).",
    "group_max": "Limit on the number of stocks to download – a full index of 500 names takes a few minutes the first time and is then cached.",
    "group_index": "Ticker used as the 'market' for comparison and beta calculations (e.g. ^GSPC = S&P 500 index, SPY = S&P 500 ETF).",
    "group_min_hist": "Stocks with less history than this (trading days) are left out of the ranking.",
    "stock_pick": "Choose one stock (or ETF) to dissect: how much of its move came from the market, from investing styles, or from the company itself.",
    "factor_window": "How many past trading days to use when estimating exposures. 756 ≈ 3 years: long enough to be stable, short enough to reflect today's business.",
    # --- macro
    "regime_scheme": "'Level': is growth/inflation above or below its own long-run average? 'Direction': is it rising or falling over the last 3 months? Direction reacts faster but flips more often.",
    "stress_q": "Financial-stress readings above this historical percentile are labelled 'Stress'.",
    "macro_pick": "Economic indicator to compare against the asset's future returns.",
    "horizon": "How many months ahead the outcome is measured. Longer horizons are smoother but there are fewer independent observations.",
    "max_lag": "How many months earlier the indicator is allowed to lead the asset's return.",
    # --- probabilities
    "fc_asset": "Asset whose next-period direction you want probabilities for.",
    "fc_h": "Forecast horizon in months.",
    "fc_band_mode": "How 'sideways' is defined. 'Volatility-scaled' adapts to the asset: a 3% move is sideways for a volatile stock but a big move for a bond fund. 'Fixed' uses the same % band for everything.",
    "fc_band": "For volatility-scaled: the sideways band is ± this many standard deviations of the horizon return (0.5 gives roughly equal thirds). For fixed mode: the band in percent.",
    "fc_price": "Use the asset's own price signals (momentum, trend, volatility, drawdown) as inputs.",
    "fc_macro": "Use macro and regime composites (growth, inflation, stress, curve) as inputs.",
    "fc_C": "Regularisation strength. Smaller = simpler, more cautious model (recommended); larger = lets the model fit noise.",
    "fc_min_train": "Months of history the model must have seen before its first out-of-sample forecast is recorded.",
    # --- general
    "refresh_full": "Re-download the full history instead of only the latest days. Slower; use if you suspect corrupted data.",
    "history_years": "How many years of price history to download for new tickers.",
    "fred_custom": "Optional: extra FRED series IDs (comma separated), e.g. UNRATE, HOUST. Find IDs on fred.stlouisfed.org.",
}

# ----------------------------------------------------------------------------------------------
# Column / metric tooltips used in tables and KPI tiles
# ----------------------------------------------------------------------------------------------
COL = {
    "MomentumPercentile": "Overall momentum rank among all ETFs on this date. 100 = strongest, 0 = weakest.",
    "MomentumScore": "Weighted blend of the percentile ranks of 6M and 12M excess return, the same with the latest month skipped, and trend smoothness.",
    "Excess_6M": "6-month return of the ETF relative to the benchmark. +5% means it beat the benchmark by about 5 percentage points.",
    "Excess_12M": "12-month return relative to the benchmark.",
    "Excess_6M_Skip1M": "6-month excess return measured up to one month ago. Skipping the latest month avoids short-term reversals that can mislead momentum signals.",
    "Excess_12M_Skip1M": "12-month excess return measured up to one month ago (the classic '12-1' momentum).",
    "RelReg_R2": "How straight the ETF's climb versus the benchmark has been over ~6 months (0–1). High = steady, low = erratic.",
    "TrendGate": "Is the price above its long-term moving average? Required for a purchase.",
    "EligibleSignal": "Passes ALL entry rules: above its trend line and positive excess return on both skip-1M windows.",
    "Vol_63D": "Annualised volatility over the last ~3 months. Higher = bumpier ride.",
    "DownsideVol_63D": "Like volatility, but counts only falling days.",
    "Ret_1M": "Return over the last ~21 trading days.", "Ret_3M": "Return over the last ~63 trading days.", "Ret_12M": "Return over the last ~252 trading days.",
    "Dist_52wHigh": "How far below its highest price of the past year (0% = at the high).",
    "Beta_1Y": "How much the stock tends to move per 1% move in the index over the last year. 1.5 = amplifies the market; 0.5 = dampens it.",
    "Corr_1Y": "How closely it moves with the index (1 = perfectly in step).",
    "MaxDD_3Y": "Worst peak-to-trough fall in the last 3 years.",
    "Vol_1Y": "Annualised volatility over the last year.",
    "PortfolioWeight": "Share of your whole portfolio.", "ActiveWeight": "Share of the momentum sleeve only.",
    "Cluster": "ETFs sharing a number move very similarly – treated as one bet.",
    "Role": "'New entry' = qualifies fresh. 'Held (hysteresis)' = already owned and allowed to stay.",
}

KPI = {
    "CAGR": "Compound annual growth rate – the steady yearly return that would produce the same final result.",
    "Volatility": "Annualised standard deviation of daily returns – the typical size of swings.",
    "Sharpe": "Return above cash per unit of volatility. Higher is better; above ~1 is good, but beware of small samples.",
    "Sortino": "Like Sharpe, but only penalises downside swings.",
    "Max drawdown": "Worst peak-to-trough fall of the portfolio value, measured on daily values.",
    "Calmar": "CAGR divided by the size of the worst drawdown.",
    "Tracking error": "Annualised volatility of the difference versus the benchmark – how differently the portfolio behaves.",
    "Information ratio": "Average yearly outperformance divided by tracking error. Above ~0.5 is respectable; needs many years to be trustworthy.",
    "Beta": "Sensitivity to the benchmark. 1 = moves with it.",
    "Upside capture": "Share of the benchmark's gains captured in months when it rose.",
    "Downside capture": "Share of the benchmark's losses suffered in months when it fell. Below 100% is good.",
    "Total return": "Cumulative gain over the whole period.", "Hit rate": "Share of days with a positive return.",
    "Best period": "Best single day.", "Worst period": "Worst single day.", "CAGR minus benchmark": "Yearly return advantage over the benchmark.",
}

# ----------------------------------------------------------------------------------------------
# Glossary
# ----------------------------------------------------------------------------------------------
GLOSSARY = {
    "q-value (false discovery)": "A p-value corrected for having looked at many things at once. With 100 comparisons, about 5 look 'significant' by pure chance at p<0.05. A q-value of 0.10 means roughly 10% of results flagged at that level are expected to be flukes.",
    "Lead / lag": "Whether one thing moves earlier than another. 'Indicator leads by 6 months' means the indicator's value six months ago lines up with the asset's return now.",
    "Adjusted close": "Closing price corrected for dividends and splits so that percentage changes equal the true total return an investor earned.",
    "Alpha": "Return not explained by exposure to the market or other known drivers. Often just noise – needs a long record and a high t-stat to be believed.",
    "Base rate": "How often something happened historically, ignoring any special information. The standard any forecasting model must beat.",
    "Basis point (bp)": "One hundredth of a percent. 10 bp = 0.10%.",
    "Benchmark": "The reference portfolio (here a global equity ETF) against which performance is judged.",
    "Beta": "How strongly an asset moves with the market. Beta 1.3 means ~1.3% move for each 1% move in the market.",
    "Bootstrap": "Re-sampling the historical data thousands of times to see how much a result could vary purely by chance.",
    "Brier score": "Average squared error of probability forecasts (lower is better). Compared with the base-rate Brier score to give a 'skill' figure.",
    "Breadth": "How many stocks in a group participate in a move. A rally led by a few stocks (narrow breadth) is more fragile.",
    "Calibration": "Do things you called 70% likely actually happen about 70% of the time?",
    "CAGR": "Compound annual growth rate.",
    "Cluster": "Group of ETFs that move almost identically, treated as a single bet.",
    "Correlation": "Number from -1 to +1 describing how closely two return series move together. It says nothing about cause.",
    "Credit spread": "Extra yield on corporate bonds over safe government bonds – a gauge of fear about company defaults.",
    "Deflated Sharpe ratio": "A Sharpe-ratio test that accounts for having tried many variants and keeping the best – corrects the 'lucky winner' problem.",
    "Dispersion": "How different individual stocks' returns are from each other. High dispersion = stock-picking matters more.",
    "Drawdown": "Fall from the previous peak, as a percentage.",
    "Excess return": "Return minus benchmark return.",
    "Factor": "A broad, persistent driver of returns shared by many stocks – e.g. value, size, momentum, profitability, investment.",
    "Hysteresis": "Using a lower bar to KEEP a holding than to BUY it, which reduces needless trading.",
    "Information ratio": "Active return per unit of tracking error.",
    "Inversion (yield curve)": "When short-term interest rates exceed long-term ones. Has preceded most US recessions, with long and variable lead times and occasional false alarms.",
    "Look-ahead bias": "Accidentally using information in a test that was not available at the time. The most common reason backtests look too good.",
    "Momentum": "The tendency of assets that did well recently (over ~3–12 months) to keep outperforming for a while – and of losers to keep lagging.",
    "Moving average": "Average price over the last N days; smooths noise and shows the trend.",
    "Multiple testing": "If you examine 100 patterns, ~5 will look 'significant' by chance. Honest analysis adjusts for this.",
    "Out-of-sample": "Data not used to build or tune a model. The only fair test of whether it works.",
    "Overfitting": "Tuning a model so closely to past noise that it fails on new data.",
    "p-value": "Probability of seeing a result at least this extreme if there were truly no effect. Small (<5%) suggests – but does not prove – a real effect.",
    "Percentile rank": "Position in a ranking from 0 (weakest) to 100 (strongest).",
    "Regime": "A recognisable economic or market environment (e.g. 'growth up, inflation down' or 'financial stress').",
    "Regularisation": "Deliberately limiting a model's flexibility to stop it fitting noise.",
    "Sharpe ratio": "Return per unit of risk (volatility).",
    "Sideways": "In this tool: a move smaller than a band around zero (see the probabilities page).",
    "Survivorship bias": "Analysing only today's survivors (current index members, ETFs that still exist) overstates past performance because failures vanished from the data.",
    "Total return": "Price change plus reinvested income (dividends).",
    "Tracking error": "Volatility of the difference between a portfolio and its benchmark.",
    "Turnover": "How much of the portfolio is traded. High turnover = high costs and (for taxable accounts) tax.",
    "UCITS": "European regulatory standard for funds sold to ordinary investors.",
    "Volatility": "Annualised standard deviation of returns.",
    "Walk-forward": "Testing a strategy by moving through time, using only the past at each step – like living through it.",
    "Yield curve": "Interest rates for different loan lengths. Normally long rates are higher than short ones.",
}

# ----------------------------------------------------------------------------------------------
# How it works
# ----------------------------------------------------------------------------------------------
HOW_DOT = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF3F8", color="#6B8BA8", fontname="Helvetica"];
  A [label="1. Free data\\nYahoo Finance prices\\nFRED economy data\\nFama-French factors"];
  B [label="2. Checks\\nquality tests\\ncurrency conversion\\nbad-tick removal"];
  C [label="3. Signals\\nmomentum rank\\ntrend filter\\nrisk & clusters"];
  D [label="4. Portfolio\\nselect, size, cap"];
  E [label="5. Honest testing\\nwalk-forward backtest\\ncosts, luck tests"];
  F [label="6. Understanding\\nregimes, macro links\\nprobabilities"];
  A -> B -> C -> D -> E; B -> F; E -> F;
}
"""

HOW_IT_WORKS = """
### The big picture
This workbench answers three different questions. Keep them separate in your mind:

1. **What is strong right now?** (*ETF Screener, Group Explorer*) – ranks ETFs or stocks by how much they have beaten a benchmark, whether they are in an uptrend, and how risky they are.
2. **Would a rule-based approach have worked?** (*Backtest Lab*) – replays history month by month, using only the information that existed at each date, charging realistic trading costs, and testing whether the result could be luck.
3. **Why does it behave this way, and what are the odds?** (*Macro & Cycles, Asset Lab, Probabilities*) – connects assets to the economic cycle, measures how much of a move came from the market, investing styles or the company itself, and produces honest probabilities for 'down / sideways / up'.

### Where the data comes from (all free)
| Data | Source | Used for |
|---|---|---|
| Prices of ETFs, stocks, indices, currencies | Yahoo Finance (via the `yfinance` library) | everything price-related |
| Economic indicators (jobs, inflation, interest rates, credit spreads, stress indices…) | FRED – Federal Reserve Bank of St. Louis | regimes, macro links, forecasts |
| Investing-style factors (market, size, value, profitability, investment, momentum) | Kenneth French Data Library | explaining *why* a stock moved |
| Index membership lists | Wikipedia | analysing the stocks inside an index |

Free data has weaknesses (occasional errors, delays, no history of delisted companies). The tool tests for the common ones and **flags problems instead of hiding them** – see *Data Center*.

### How the momentum score is built
For every ETF on a given date the tool measures:
- 6-month and 12-month **excess return** versus the benchmark,
- the same two measures **skipping the most recent month** (short-term moves tend to partly reverse),
- the **smoothness** of the climb relative to the benchmark (R² of a trend line),
then converts each into a percentile rank among all ETFs and blends them (weights 20/30/15/20/15%). To be *bought* an ETF must also be above its long-term moving average and have positive skip-1-month excess returns. Holdings are kept until they fall below a lower 'exit' rank (hysteresis).

### What stops the tool fooling you
- **No look-ahead:** each decision only sees data up to its own date. A built-in test scrambles the future and confirms past decisions do not change.
- **Realistic timing and costs:** trades happen after the signal (default one day later); costs are charged on everything bought and sold.
- **Luck tests:** bootstrap confidence intervals, probabilistic and *deflated* Sharpe ratios, and permutation tests for regime effects.
- **Out-of-sample scoring:** probability models are always compared with a naive base rate, and ignored if they do not beat it.
- **Self-test:** the *Data Center* page can verify the calculation engine against known answers on your own computer.
"""

# ----------------------------------------------------------------------------------------------
# Tutorials  (title, markdown)
# ----------------------------------------------------------------------------------------------
TUTORIALS: list[tuple[str, str]] = [
("0 · Comparing assets with the economy", """
**Goal:** see which ETFs, stocks and sectors move with an economic measure – without fooling yourself.

1. Open **Compare with Economy**, pick an indicator (try *Growth* or *Inflation*).
2. Leave **Add sector ETFs** on. Sectors have decades of history, so they are the most reliable evidence, and they show *where* in the market the link lives.
3. Read the **Verdict** column, not the correlation. *Supported* means: significant after correcting for the number of comparisons, same direction in both halves of history, and same direction at neighbouring lags. *Suggestive only* means treat as a hint.
4. Check the **Lag map**: a real link looks like a smooth block of one colour, not a single bright cell.
5. Check **Rolling stability**: if the sign flips over time, don't build a decision on it.

**Rule of thumb:** the more indicators and lags you browse, the more flukes you will find. A link is worth acting on only if it also has an economic reason you can explain in a sentence.
"""),
("1 · Ten-minute tour", """
**Goal:** go from a fresh install to a first result.

1. Open **Data Center** → click **Update ETF prices**. The first run downloads about 12 years of history for every ETF in your universe (a few minutes). Later updates only fetch the new days.
2. Still in Data Center, click **Update economic data** (FRED) and, if you want to analyse stocks, **Update style factors**.
3. Open **ETF Screener**. You will see the current ranking and the model portfolio. Hover over any column title or setting to read what it means.
4. Open **Backtest Lab** → *Run backtest*. Look at the *Is it luck?* panel before you look at the return figure.
5. Open **Macro & Cycles** to see which economic regime we are in and how assets historically behaved in it.
6. Open **Probabilities** to get Down / Sideways / Up odds for an asset – and check whether the model has earned the right to be believed.

**Golden rule:** treat every output as a *question to investigate*, not an instruction to trade.
"""),
("2 · What is momentum and why might it work?", """
**Momentum** means recent winners tend to keep winning for a while, and losers keep losing. It is one of the most studied patterns in finance (academic work from the early 1990s onward, e.g. Jegadeesh & Titman 1993, and later added as a standard 'factor').

**Possible reasons it exists** (researchers disagree on which dominate):
- Investors under-react to news at first and pile in later (herding).
- Trends in earnings, money flows, and economic conditions are persistent.
- Institutions cannot all rebalance at once.

**Why it can fail**
- *Momentum crashes*: after sharp market falls, previous losers can rebound violently while winners fall.
- It becomes crowded; it costs money to trade; and past premiums need not repeat.

**Practical consequence:** this tool uses momentum together with a trend filter and spreads risk across clusters so one theme cannot dominate. The pattern has historically been *modest and inconsistent* – which is why the Backtest Lab spends so much effort telling you whether a result could be luck.
"""),
("3 · Reading the ETF Screener", """
Each row is one ETF on the chosen date.

- **Momentum percentile** – the headline rank (100 = strongest of the universe).
- **Excess 6M / 12M** – how much it beat the benchmark. *Skip-1M* versions ignore the latest month.
- **Trend R²** – steadiness of the climb (1 = ruler-straight).
- **Trend gate** – above its moving average? If not, it cannot be bought.
- **Volatility** – bumpiness. Used to size positions.
- **Cluster** – ETFs sharing a number are look-alikes; the model picks from different clusters first.
- **Role** – *New entry* (qualifies fresh) vs *Held (hysteresis)* (already owned).

The **Decision audit** table accounts for *every* ETF and states why it was or was not selected – the reasons are rule-based, so you can verify them.
"""),
("4 · Benchmarks, excess returns and percentiles", """
A fund that rose 12% sounds good – until you learn the global market rose 18%. That is why everything is measured **relative to a benchmark**.

- **Excess return** = (1 + ETF return) ÷ (1 + benchmark return) − 1.
- **Percentile rank** converts a raw number into a position in the league table (0–100). Ranks are *relative*: even in a falling market someone is top.
- This is why a **trend filter** and positive-excess requirements are also used – to avoid buying the 'best of a bad lot'.

Currency matters too: an ETF quoted in dollars rose partly because the dollar rose. The tool converts every price into your chosen base currency (GBP by default) before comparing.
"""),
("5 · What a backtest can and cannot tell you", """
A backtest is a *simulation of the past*. Treat it as a smoke test: it can reveal that an idea is **bad**, but can almost never prove an idea is **good**.

**Things the tool does to keep it honest**
- Decisions use only data up to their date (tested automatically).
- Trades happen *after* the signal (default 1 day) and cost money (both buy and sell).
- Portfolio value is marked **every day**, so drawdowns are real.
- It compares against the benchmark *and* an equal-weight basket of the whole universe, so you can see whether momentum selection adds anything beyond 'being in the universe'.

**Things it cannot fix**
- **Survivorship bias** – the ETF list contains funds that exist today. Funds that closed are missing, flattering history.
- **Overfitting** – trying many settings and showing the best. Use the **Robustness grid**: a real effect works across *neighbouring* settings; the *Deflated Sharpe* column discounts for the number of tries.
- **Regime change** – the future need not resemble the sample.
- **Small samples** – 5 years of monthly decisions is only 60 observations.

**How to read the "Is it luck?" panel:** the interval shows the plausible range of the true monthly out-performance. If it includes 0, you cannot rule out luck.
"""),
("6 · Risk metrics in plain English", """
| Metric | Plain meaning | Rule of thumb |
|---|---|---|
| **Volatility** | typical size of swings | global equities ≈ 15% a year |
| **Max drawdown** | worst fall from a peak | ask yourself: could I sit through this? |
| **Sharpe** | reward per unit of risk | >1 good; <0.5 mediocre; noisy with short data |
| **Sortino** | like Sharpe, only counts bad swings | |
| **Calmar** | yearly return ÷ worst drawdown | |
| **Tracking error** | how differently you behave vs. the benchmark | higher = more 'active risk' |
| **Information ratio** | out-performance ÷ tracking error | >0.5 respectable, rarely sustained |
| **Up/Down capture** | share of benchmark's gains/losses you got | want up > down |
| **Beta** | sensitivity to the benchmark | 1 = moves with it |

**Drawdowns matter more than most people expect:** a 50% fall needs a 100% gain to recover.
"""),
("7 · Correlation, clusters and diversification", """
Holding five ETFs that all follow US technology is *one* bet, not five. **Correlation** measures how similarly two things move (−1 opposite, 0 unrelated, +1 identical).

The tool groups ETFs whose returns are highly correlated into **clusters**, picks the best ETF of each cluster first, and caps the share any cluster may hold. The *Risk & Overlap* view in the ETF Screener shows the correlation heat-map.

Beware: correlations are **not constant**. In crises, assets that normally move independently often fall together – diversification works least when you need it most.
"""),
("8 · Economic cycles and regimes 101", """
Economies move through phases, and different assets tend to do better in different phases. The tool labels each month using open data:

| Dimension | Question | How it is measured |
|---|---|---|
| **Growth / inflation quadrant** | Is growth above or below normal? Is inflation above or below normal? | composites of jobs, production, retail sales, claims… / CPI, core CPI, core PCE, market-implied inflation |
| **Market stress** | Are financial markets under strain? | Chicago Fed index, St. Louis stress index, credit spreads, VIX |
| **Yield curve** | Is the curve inverted? | 10-year minus 3-month Treasury yield |
| **Policy stance** | Is the Fed tightening or easing? | 6-month change in the policy rate |

The four growth/inflation states: **Goldilocks** (growth up, inflation down), **Overheating** (both up), **Stagflation** (growth down, inflation up), **Slowdown** (both down).

**Important:** labels use only information that had been *published* at each date (CPI for March is not known in March). Still, free data are later revised, so real-time accuracy was somewhat lower than the charts imply.
"""),
("9 · Macro indicators explained", """
Open **Macro & Cycles → Indicator guide** for a one-line explanation of every series. Some headline ideas:

- **Yield curve (10y − 3m)** negative → markets expect rate cuts, often because trouble is expected. It has preceded most US recessions since the 1960s, typically by many months to two years; timing is unreliable and there have been false alarms.
- **Credit spreads** widen when investors fear defaults – often *before* equities fall sharply.
- **Jobless claims** are the earliest weekly signal of layoffs.
- **Inflation expectations / real yields** influence valuations: higher real yields squeeze long-duration assets such as growth stocks and long bonds.
- **Financial conditions indices** (NFCI, St. Louis stress) summarise credit, leverage and risk in one number.

A single indicator is a weak guide; the tool combines groups of indicators into composites to reduce noise.
"""),
("10 · Correlation is not causation", """
Finding that an indicator and an asset move together does **not** mean one causes the other – both may react to a third thing, or it may be coincidence.

**How this tool guards against fooling you**
- *Lead-lag* tests use **HAC (Newey-West) statistics**, which correct for overlapping return windows that otherwise make links look far stronger than they are.
- *Regime tables* show **independent observations**: 36 monthly points in a 3-month-ahead study are only ~12 independent windows. Confidence labels say 'Low' when evidence is thin.
- *Permutation p-values* keep regimes' persistence, so slow-moving regimes cannot manufacture false significance.
- *Out-of-sample R²*: an indicator is only useful if it improves forecasts on data the model had not seen.
- With dozens of indicators and assets, **some links will look significant by chance** (multiple testing). Prefer links with an economic story, that appear in several assets or periods.
"""),
("11 · Probabilities: down, sideways, up", """
The **Probabilities** page gives three views of the odds for the next N months:

1. **Base rate** – how often each outcome happened over the whole history. This is the benchmark.
2. **Regime-conditional** – the same, restricted to months resembling today's economic regime (with confidence ranges).
3. **Model** – a regularised statistical model using price signals and macro composites.

**Is the model any good?** It is tested **walk-forward**: for each past month it is trained only on outcomes already known, then asked to forecast. Its **Brier score** (squared error of probabilities) is compared with the base rate. The page prints a plain verdict: *beats base rate*, *no reliable edge*, or *worse*. In most cases and for most assets the honest verdict will be 'no reliable edge' – markets are hard to predict. That is a valuable result: it protects you from false confidence.

**Calibration chart:** if the model says 60% 'up', roughly 60% of such months should have been up. Points on the diagonal mean trustworthy probabilities.

**How to use it:** probabilities are for *sizing and risk awareness* (for example, a 55% chance of an up-move is barely better than a coin-flip), not for predicting a specific outcome.
"""),
("12 · Analysing a group of stocks", """
In **Group Explorer** you can load an index (S&P 500, Nasdaq-100, FTSE 100, DAX 40, CAC 40, Dow 30), a sector, or your own list.

- **Ranking table** – same momentum model as ETFs, with each stock compared to the index.
- **Breadth** – what share of members are above their moving averages or making new highs. Strong index + weak breadth = narrow, fragile leadership.
- **Equal-weight vs index** – when the equal-weight basket lags the index, a few giant companies are doing the work (concentration).
- **Dispersion & correlation** – when correlations are very high, stocks move as one block (macro-driven); when low, company-specific factors dominate.
- **Contribution** – which stocks drove the group's return.

⚠️ **Survivorship warning:** index lists are *today's* members. Companies that fell out of the index years ago are absent, which flatters the group's history. Use history for understanding behaviour, not for promising future returns.
"""),
("13 · Why did this stock move? Beta, factors and alpha", """
Every stock's return can be split into pieces:

- **Market (beta):** the part that comes from the overall market moving.
- **Style factors:** size (small vs large), value (cheap vs expensive), profitability, investment, and momentum.
- **Alpha / residual:** what is left – the company's own story *plus noise*.

The **Asset Lab** shows the split for any window. A stock with R² of 0.7 is mostly driven by general forces; one with R² of 0.2 is dominated by company-specific news.

Factor data are US-based. For non-US stocks they are only a rough lens; the tool warns you. **Alpha** estimates are noisy – look at the t-stat: below ~2 you cannot tell it from zero.
"""),
("14 · From signals to decisions: a sensible checklist", """
Before acting on anything here, ask:

1. **Data:** is the data fresh? Any red flags in the Data Center quality table?
2. **Evidence:** does the backtest beat both the benchmark *and* the equal-weight universe? Is the 'luck' interval clear of zero? Does the robustness grid agree across neighbouring settings?
3. **Mechanism:** can I explain in one sentence *why* it might work?
4. **Costs & taxes:** after realistic trading costs and tax in my account type, is it still worthwhile?
5. **Size:** what happens to my life if this holding falls 30–50%? Limit the share of wealth at risk.
6. **Simplicity:** would a broad low-cost index fund do almost as well with none of the effort?
7. **Second opinion:** verify key numbers independently (issuer factsheets, a second data source). Consider a regulated adviser for big decisions.

This software cannot know your goals, taxes, time horizon or risk tolerance.
"""),
]

PITFALLS = [
    ("Survivorship bias", "The universe only contains funds/stocks that exist today. Failures are invisible, so history looks better than reality."),
    ("Look-ahead bias", "Using information that was not yet public. Guarded against with publication lags, execution lag and an automated scramble-the-future test."),
    ("Overfitting / data mining", "Trying many settings and reporting the best. Use the robustness grid and the deflated Sharpe ratio."),
    ("Small samples", "A few years of monthly decisions is not much evidence. Wide confidence intervals are normal."),
    ("Data revisions", "Free macro data are revised after first publication. Backtests using revised numbers are slightly too optimistic."),
    ("Regime change", "Relationships that held in one era (e.g. inflation–equity links) can flip in another."),
    ("Costs and slippage", "Real costs include spreads, platform fees, FX conversion and taxes. Test higher cost assumptions."),
    ("Currency effects", "Returns depend on the currency you measure in. The tool converts to your base currency, but listings' currencies are detected automatically – verify in Data Center."),
    ("Multiple testing", "Looking at many assets/indicators guarantees some false discoveries."),
    ("Overconfidence", "A precise-looking number is not an accurate number."),
]

FAQ = [
    ("Why does the tool sometimes say 'no reliable edge'?", "Because most apparent patterns in markets are not strong enough to survive honest, out-of-sample testing. Reporting that is the tool doing its job."),
    ("Can I trust Yahoo Finance data?", "It is generally good for major ETFs and stocks but occasionally contains errors (bad prices, wrong currencies, missing days). The tool screens for these and excludes failing series; still verify anything important."),
    ("Why GBP?", "The original workbook targets UK investors holding London-listed ETFs. Change the base currency in the settings."),
    ("The numbers differ from v7.8. Why?", "Costs are now charged on both buy and sell legs (v7.8 charged half), trades happen one day after the signal, drawdowns use daily values, and held positions are no longer dropped by the entry filter. See the audit report."),
    ("How much history do I need?", "For momentum ranking: about 1 year per ETF. For regime and probability studies: ideally 15+ years of monthly data, which covers several cycles."),
    ("Can this predict crashes?", "No. It can show which conditions historically preceded weak returns, and give odds – with large uncertainty."),
    ("What does 'point-in-time' mean?", "Using only information that was available on the date. It is the key to a fair backtest."),
    ("Is anything uploaded anywhere?", "No. The app runs on your computer; it only downloads public data and stores results in the 'data', 'logs' and 'reports' folders."),
]
