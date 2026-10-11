"""Orakel-Tests: Gauß-Prozess gegen scikit-learn, Batch-Regeln gegen Handformeln, Studie gegen eigene Neuberechnung."""

import numpy as np
import pytest

import bbo_algorithm as A
import bbo_constants as C
import bbo_evaluation as ev


def test_gp_posterior_equals_scikit_learn_for_fixed_hyperparameters():
    sk = pytest.importorskip("sklearn.gaussian_process")
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    rng = np.random.default_rng(11)
    for trial in range(6):
        d = 1 + trial % 3
        X = rng.random((9, d))
        y = np.sin(3 * X[:, 0]) + 0.2 * X[:, -1]
        ls = rng.uniform(0.3, 1.5, d)
        var = 1.7
        gp = A.GaussianProcess()
        gp.y_mean, gp.y_std = float(y.mean()), float(y.std())
        gp.lengthscales, gp.variance = ls, var
        gp.condition(X, y)
        Xs = rng.random((5, d))
        mu, sd = gp.predict(Xs)
        kern = ConstantKernel(var, "fixed") * Matern(length_scale=ls, nu=2.5, length_scale_bounds="fixed")
        z = (y - y.mean()) / y.std()
        ref = sk.GaussianProcessRegressor(kern, alpha=1e-6 + 1e-8, optimizer=None).fit(X, z)
        m_ref, s_ref = ref.predict(Xs, return_std=True)
        assert np.allclose(mu, m_ref * y.std() + y.mean(), atol=1e-6), trial
        assert np.allclose(sd, s_ref * y.std(), atol=1e-4), trial


def test_believer_update_equals_posterior_with_an_added_noise_free_observation():
    """Kriging Believer = Bedingung auf (x, mu(x)): Mittelwert unverändert, Varianz wie nach einer echten Beobachtung (Handformel s'^2 = s^2 − k_x^2/k_xx)."""
    rng = np.random.default_rng(4)
    X = rng.random((8, 2))
    y = np.cos(4 * X[:, 0]) + X[:, 1]
    gp = A.GaussianProcess().fit(X, y)
    x = np.array([[0.31, 0.62]])
    Xs = rng.random((6, 2))
    mu0, sd0 = gp.predict(Xs)
    mx, _ = gp.predict(x)
    g2 = gp.copy_with(np.vstack([X, x]), np.append(y, mx))
    mu1, sd1 = g2.predict(Xs)
    assert np.allclose(mu1, mu0, atol=1e-6 * (np.abs(mu0).max() + 1))                # Bedingen auf den eigenen Mittelwert ändert den Mittelwert nirgends
    # Varianz nach der Beobachtung: Vorher-Kovarianz zwischen Xs und x aus der posterior-Stichprobe der Handformel
    S = gp.sample(np.vstack([Xs, x]), np.random.default_rng(0), n=20000)
    cov = np.cov(S.T)
    k_sx, k_xx = cov[:-1, -1], cov[-1, -1]
    expected_var = sd0 ** 2 - k_sx ** 2 / k_xx
    assert np.allclose(sd1 ** 2, expected_var, atol=0.08 * (sd0 ** 2).max() + 1e-4)


def test_expected_improvement_equals_numerical_integration():
    from scipy.integrate import quad
    from scipy.stats import norm
    for mu, sd, best in [(0.3, 0.7, 0.0), (-0.5, 0.2, 0.1), (1.0, 2.0, 0.5)]:
        ref = quad(lambda f: max(best - f, 0.0) * norm.pdf(f, mu, sd), mu - 12 * sd, mu + 12 * sd)[0]
        assert A.expected_improvement(np.array([mu]), np.array([sd]), best)[0] == pytest.approx(ref, abs=1e-8)


def test_sign_test_matches_the_exact_binomial_by_hand():
    from math import comb
    p_hand = 2 * sum(comb(20, k) for k in range(0, 5)) / 2 ** 20            # 4 von 20 gegen 16: zweiseitig
    from scipy.stats import binomtest
    assert binomtest(4, 20, 0.5).pvalue == pytest.approx(p_hand)
    assert p_hand == pytest.approx(0.01181793, abs=1e-6)


def test_study_paired_stats_recompute_from_the_raw_runs():
    raw = ev.study()["runs"]
    for prob, v in (("hartmann", "kb_q4"), ("branin", "topq_q8")):
        a = np.array(raw[prob][v]["best"])[:, C.TOTAL[prob] - 1]
        b = np.array(raw[prob]["seq"]["best"])[:, C.TOTAL[prob] - 1]
        s = ev.paired_stats(prob, v, "seq", by="evaluations")
        assert s["better"] == int((a < b - 1e-9).sum()) and s["worse"] == int((a > b + 1e-9).sum())
        assert s["better"] + s["worse"] + s["equal"] == 20


def test_study_is_complete_and_monotone_and_shares_start_values():
    raw = ev.study()["runs"]
    for p in C.PROBLEMS:
        assert set(raw[p]) == set(C.VARIANTS)
        n0 = C.N_START[p]
        first = None
        for v, r in raw[p].items():
            arr = np.array(r["best"])
            assert arr.shape == (20, C.TOTAL[p]), (p, v)
            assert np.all(np.diff(arr, axis=1) <= 1e-12)
            assert np.all(arr - C.GLOBAL_MIN[p] >= -1e-6)
            start = arr[:, n0 - 1]                                        # bester Wert des Startdesigns: gleich für alle Varianten eines Seeds
            if first is None:
                first = start
            assert np.allclose(start, first), (p, v)
