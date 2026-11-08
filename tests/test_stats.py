import math
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from workbench import stats as S


def test_max_drawdown_known():
    w = pd.Series([1.0, 1.2, 0.9, 1.0, 1.3, 0.65])
    assert S.max_drawdown(w) == pytest.approx(0.65 / 1.3 - 1)
    # starting below 1 counts the initial wealth of 1.0
    assert S.max_drawdown(pd.Series([0.8, 0.9])) == pytest.approx(-0.2)


def test_drawdown_table_known():
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    w = pd.Series([1, 1.1, 0.88, 0.99, 1.1, 1.2, 0.6, 0.7], index=idx)
    t = S.drawdown_table(w, top=5)
    assert t.iloc[0]["Depth"] == pytest.approx(0.6 / 1.2 - 1)
    assert t.iloc[0]["Peak date"] == idx[5] and pd.isna(t.iloc[0]["Recovery date"])
    assert t.iloc[1]["Depth"] == pytest.approx(0.88 / 1.1 - 1)
    assert t.iloc[1]["Recovery date"] == idx[4]


def test_performance_constant_return():
    r = pd.Series(0.01, index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    m = S.performance_metrics(r, ppy=12)
    assert m["CAGR"] == pytest.approx(1.01**12 - 1)
    assert m["Max drawdown"] == 0 and m["Hit rate"] == 1


def test_ols_matches_lstsq_and_white_se():
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 2))
    y = 0.5 + X @ np.array([1.0, -2.0]) + rng.normal(size=n) * (1 + np.abs(X[:, 0]))
    res = S.ols_hac(y, X, lags=0)
    Xc = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]
    assert np.allclose(res.coef, beta)
    # independent White HC0
    u = y - Xc @ beta
    bread = np.linalg.inv(Xc.T @ Xc)
    meat = sum(np.outer(Xc[i], Xc[i]) * u[i] ** 2 for i in range(n))
    se = np.sqrt(np.diag(bread @ meat @ bread))
    assert np.allclose(res.se, se, rtol=1e-9)


def test_newey_west_matches_explicit_loop():
    rng = np.random.default_rng(2)
    n, L = 200, 5
    x = rng.normal(size=n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.6 * e[t - 1] + rng.normal()
    y = 1 + 0.3 * x + e
    res = S.ols_hac(y, x, lags=L)
    X = np.column_stack([np.ones(n), x])
    b = np.linalg.solve(X.T @ X, X.T @ y)
    u = y - X @ b
    # explicit sums over time (independent of the vectorised implementation)
    S0 = sum(np.outer(X[t], X[t]) * u[t] ** 2 for t in range(n))
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        for t in range(l, n):
            G = np.outer(X[t], X[t - l]) * u[t] * u[t - l]
            S0 += w * (G + G.T)
    Bi = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(Bi @ S0 @ Bi))
    assert np.allclose(res.se, se, rtol=1e-9)


def test_wilson_known_value():
    lo, hi = S.wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=2e-4) and hi == pytest.approx(0.9433, abs=2e-4)
    assert S.wilson_interval(0, 5)[0] == 0.0


def test_multinomial_logit_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    n = 400
    X = rng.normal(size=(n, 4))
    logits = np.column_stack([X[:, 0], -X[:, 0] + 0.5 * X[:, 1], 0.3 * X[:, 2]])
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=pi) for pi in p])
    for C in (0.05, 0.3, 3.0):
        m = S.MultinomialLogit(C=C).fit(X, y, 3)
        Z = (X - X.mean(0)) / X.std(0)
        sk = LogisticRegression(C=C, tol=1e-10, max_iter=5000).fit(Z, y)
        assert np.allclose(m.predict_proba(X), sk.predict_proba(Z), atol=2e-4), C
        assert m.converged_


def test_psr_formula_normal_case():
    # for normal returns skew=0, kurt=3: PSR = Phi((SR-SR*)*sqrt(n-1)/sqrt(1+SR^2/2))
    sr, n = 0.2, 60
    expect = sps.norm.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + sr**2 / 2))
    assert S.probabilistic_sharpe(sr, n, 0.0, 3.0) == pytest.approx(expect)


def test_expected_max_sharpe_monte_carlo():
    rng = np.random.default_rng(4)
    N, V = 100, 0.01
    mc = rng.normal(0, math.sqrt(V), size=(20000, N)).max(1).mean()
    assert S.expected_max_sharpe(N, V) == pytest.approx(mc, rel=0.04)


def test_deflated_sharpe_false_positive_rate():
    """Best of 50 pure-noise strategies should NOT usually look 'significant' after deflation."""
    rng = np.random.default_rng(5)
    hits_dsr = hits_psr = 0
    reps = 150
    for _ in range(reps):
        R = rng.normal(0, 0.04, size=(120, 50))
        srs = R.mean(0) / R.std(0, ddof=1)
        best = R[:, srs.argmax()]
        r = pd.Series(best)
        sk, ku = sps.skew(best, bias=False), sps.kurtosis(best, fisher=False, bias=False)
        hits_psr += S.probabilistic_sharpe(srs.max(), 120, sk, ku, 0.0) > 0.95
        hits_dsr += S.deflated_sharpe(r, 50, np.var(srs, ddof=1))["dsr"] > 0.95
    assert hits_psr / reps > 0.5          # naive test is fooled most of the time
    assert hits_dsr / reps < 0.12         # deflated test controls the false-positive rate


def test_bootstrap_ci_coverage():
    rng = np.random.default_rng(6)
    cover = 0
    reps = 120
    for i in range(reps):
        x = rng.normal(0.0, 1.0, 120)
        ci = S.bootstrap_ci(x, n_boot=400, mean_block=4, seed=i)
        cover += ci["lo"] <= 0 <= ci["hi"]
    assert 0.88 <= cover / reps <= 0.99


def test_circular_shift_controls_false_positives_for_persistent_labels():
    rng = np.random.default_rng(7)
    n, reps, rej_c, rej_t = 240, 150, 0, 0
    for i in range(reps):
        # persistent regime labels (AR-like), returns independent of labels
        lab = np.zeros(n, dtype=int)
        for t in range(1, n):
            lab[t] = lab[t - 1] if rng.random() < 0.93 else 1 - lab[t - 1]
        x = rng.normal(size=n)
        if lab.sum() < 10 or (1 - lab).sum() < 10:
            continue
        rej_c += S.circular_shift_test(x, lab, 1, n_perm=300, seed=i)["p"] < 0.05
        rej_t += sps.ttest_ind(x[lab == 1], x[lab == 0]).pvalue < 0.05
    assert rej_c / reps < 0.10            # permutation test is close to nominal 5%
    # plain t-test is NOT guaranteed here; we only record it is not better calibrated by construction
    assert rej_c <= rej_t + 5


def test_scores():
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y = np.array([0, 1])
    assert S.brier_score(P, y) == pytest.approx(((0.09 + 0.04 + 0.01) + (0.01 + 0.04 + 0.01)) / 2)
    assert S.log_loss(P, y) == pytest.approx(-(math.log(0.7) + math.log(0.8)) / 2)
