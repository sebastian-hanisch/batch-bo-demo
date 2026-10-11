"""Einheiten des Kerns an Handrechnungen und Mini-Instanzen (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import bbo_algorithm as A
import bbo_constants as C
import bbo_scenario as S


def test_matern_kernel_hand_values():
    x = np.array([[0.0]])
    assert A.matern52(x, x, np.array([1.0]), 2.0)[0, 0] == pytest.approx(2.0)
    k = A.matern52(x, np.array([[1.0]]), np.array([1.0]), 1.0)[0, 0]
    assert k == pytest.approx((1 + np.sqrt(5) + 5 / 3) * np.exp(-np.sqrt(5)))


def test_expected_improvement_hand_values():
    assert A.expected_improvement(np.array([0.0]), np.array([1.0]), 0.0)[0] == pytest.approx(0.3989422804)
    assert A.expected_improvement(np.array([-1.0]), np.array([1e-12]), 0.0)[0] == pytest.approx(1.0)
    assert A.expected_improvement(np.array([1.0]), np.array([1e-12]), 0.0)[0] == pytest.approx(0.0, abs=1e-12)


def test_gp_interpolates_noise_free_data(mini):
    X, y = mini
    gp = A.GaussianProcess().fit(X, y)
    mu, sd = gp.predict(X)
    assert np.allclose(mu, y, atol=1e-3)
    assert np.all(sd < 1e-2)


def test_conditioning_on_a_fantasy_removes_uncertainty_there(mini):
    X, y = mini
    gp = A.GaussianProcess().fit(X, y)
    x = np.array([[0.4]])
    mu, sd = gp.predict(x)
    g2 = gp.copy_with(np.vstack([X, x]), np.append(y, mu))
    mu2, sd2 = g2.predict(x)
    assert sd2[0] < 1e-2 < sd[0] or sd2[0] < sd[0]
    assert mu2[0] == pytest.approx(mu[0], abs=1e-3)               # Kriging Believer ändert den Mittelwert an der gewählten Stelle nicht
    # und die Hyperparameter bleiben
    assert np.array_equal(g2.lengthscales, gp.lengthscales)


def test_posterior_samples_have_the_posterior_mean_and_pass_through_data(mini):
    X, y = mini
    gp = A.GaussianProcess().fit(X, y)
    Xs = np.array([[0.1], [0.4], [0.9]])
    S_ = gp.sample(Xs, np.random.default_rng(0), n=4000)
    mu, sd = gp.predict(Xs)
    assert np.allclose(S_.mean(0), mu, atol=0.05 * (np.abs(mu).max() + 1))
    assert np.allclose(S_.std(0), sd, rtol=0.2, atol=0.01)
    at_data = gp.sample(X, np.random.default_rng(1), n=3)
    assert np.allclose(at_data, y, atol=1e-2)


def test_batch_has_q_points_inside_the_cube_for_every_rule():
    rng = np.random.default_rng(2)
    X = rng.random((10, 3))
    y = np.array([S.hartmann6(np.concatenate([x, x])) for x in X])
    gp = A.GaussianProcess().fit(X, y)
    for m in C.METHODS:
        B = A.select_batch(gp, X, y, 4, np.random.default_rng(5), m)
        assert B.shape == (4, 3) and B.min() >= 0 and B.max() <= 1, m


def test_believer_batch_points_are_spread_wider_than_the_naive_top_q():
    spreads = {"topq": [], "kb": []}
    for seed in range(6):
        rng = np.random.default_rng(seed)
        X = rng.random((12, 2))
        y = np.array([S.branin(x) for x in X])
        gp = A.GaussianProcess().fit(X, y)
        for m in spreads:
            B = A.select_batch(gp, X, y, 4, np.random.default_rng(seed + 50), m)
            d = np.sqrt(((B[:, None] - B[None]) ** 2).sum(-1))
            spreads[m].append(d[np.triu_indices(4, 1)].mean())
    assert np.median(spreads["kb"]) > 2 * np.median(spreads["topq"])


def test_q1_batch_search_equals_a_sequential_loop_in_length_and_rounds():
    f = lambda x: float(((x - 0.3) ** 2).sum())
    X, y, rounds, spread = A.batch_search(f, 2, 4, 12, 1, np.random.default_rng(0), "kb")
    assert len(y) == 12 and rounds == 8 and spread == []


def test_last_round_may_be_smaller_and_start_design_is_shared():
    f = lambda x: float(((x - 0.3) ** 2).sum())
    X0 = np.random.default_rng(9).random((5, 2))
    X, y, rounds, spread = A.batch_search(f, 2, 5, 14, 4, np.random.default_rng(1), "cl", X0=X0)
    assert len(y) == 14 and rounds == 3                              # 4 + 4 + 1
    assert np.array_equal(X[:5], X0)
    assert len(spread) == 2                                          # die Runde mit einem Punkt hat keine Streuung


def test_known_function_values():
    xstar = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    assert S.hartmann6(xstar) == pytest.approx(-3.32237, abs=1e-4)
    assert S.ackley6(np.full(6, 0.5)) == pytest.approx(0.0, abs=1e-9)
    # Branin: das Minimum bei (−π, 12,275) liegt auf [0,1]^2 bei u = (−π + 5)/15, v = 12,275/15
    assert S.branin(np.array([(5 - np.pi) / 15, 12.275 / 15])) == pytest.approx(0.397887, abs=1e-5)
