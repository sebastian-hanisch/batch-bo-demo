"""Testfunktionen auf dem Einheitswürfel [0, 1]^d (Minimierung, rauschfrei)."""

import numpy as np

_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650],
                      [4047, 8828, 8732, 5743, 1091, 381]])
_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])


def branin(x):
    """Branin auf [0, 1]^2 (x1 = −5 + 15 u, x2 = 15 v); drei gleichwertige Minima, Wert 0,397887."""
    u, v = float(x[0]), float(x[1])
    x1, x2 = -5.0 + 15.0 * u, 15.0 * v
    b, c, r, s, t = 5.1 / (4 * np.pi ** 2), 5.0 / np.pi, 6.0, 10.0, 1.0 / (8 * np.pi)
    return float((x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)


def hartmann6(x):
    """Hartmann6 (Dixon und Szegö), Minimum −3,32237."""
    z = np.asarray(x, float)[:6]
    return -float((_ALPHA * np.exp(-(_A * (z - _P) ** 2).sum(1))).sum())


def ackley6(x):
    """Ackley, z = 8x − 4, alle Eingaben wirksam; Minimum 0 bei x = 0,5."""
    z = 8.0 * np.asarray(x, float) - 4.0
    return float(-20.0 * np.exp(-0.2 * np.sqrt((z ** 2).mean())) - np.exp(np.cos(2.0 * np.pi * z).mean()) + 20.0 + np.e)


def objective(problem):
    return {"branin": branin, "hartmann": hartmann6, "ackley": ackley6}[problem]
