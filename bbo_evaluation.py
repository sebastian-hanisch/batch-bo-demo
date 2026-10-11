"""Auswertung: ein Lauf je Variante, Verlauf über Auswertungen und über Runden, vorgerechnete Studie mit gepaarten Läufen und Vorzeichentest."""

import json
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

import bbo_algorithm as A
import bbo_constants as C
import bbo_scenario as S

_HERE = os.path.dirname(os.path.abspath(__file__))


def variant_of(method, q):
    return "seq" if q == 1 else f"{method}_q{q}"


def split_variant(variant):
    """'kb_q4' -> ('kb', 4); 'seq' -> ('seq', 1)."""
    if variant == "seq":
        return "seq", 1
    m, q = variant.split("_q")
    return m, int(q)


@dataclass(frozen=True)
class Settings:
    problem: str = "branin"
    q: int = 4
    method: str = "kb"
    seed: int = C.DEFAULT_SEED


@dataclass
class Run:
    variant: str
    best_so_far: list          # bester Wert nach jeder Auswertung (Start eingeschlossen)
    rounds: int
    spread: list               # mittlerer Abstand der Punkte einer Runde (bei q = 1 leer)
    seconds: float
    X: np.ndarray = field(repr=False, default=None)
    round_of: list = field(default_factory=list)       # Runde je Auswertung (0 = Startdesign)


def run_one(problem, method, q, seed, total=None):
    f = S.objective(problem)
    total = total or C.TOTAL[problem]
    rng = np.random.default_rng(seed + 1000 * (C.METHODS.index(method) + 1 if q > 1 else 0) + 17 * q)
    t0 = time.perf_counter()
    X0 = np.random.default_rng(seed).random((C.N_START[problem], C.DIM[problem]))     # Startdesign nur vom Seed abhängig: gepaarter Vergleich
    X, y, rounds, spread = A.batch_search(f, C.DIM[problem], C.N_START[problem], total, q, rng, method if q > 1 else "kb", X0=X0)
    n0 = C.N_START[problem]
    round_of = [0] * n0 + [1 + i // q for i in range(len(y) - n0)]
    return Run(variant_of(method, q), [float(v) for v in np.minimum.accumulate(y)], rounds, spread, time.perf_counter() - t0, X, round_of)


def regret(problem, values):
    return np.asarray(values, float) - C.GLOBAL_MIN[problem]


def best_after_rounds(best, problem, q, r):
    """Bester Wert nach r Runden (0 = nur Startdesign) aus dem Verlauf nach Auswertungen; nach der letzten Runde bleibt der Endwert."""
    idx = min(C.N_START[problem] + r * q, len(best)) - 1
    return best[idx]


# --- Vorgerechnete Studie ------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def study():
    with open(os.path.join(_HERE, "precomputed_study.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _best(problem, variant):
    return np.array(study()["runs"][problem][variant]["best"])             # Läufe x Auswertungen


def rounds_in_budget(problem, variant):
    _, q = split_variant(variant)
    return int(np.ceil((C.TOTAL[problem] - C.N_START[problem]) / q))


def regret_at_evaluations(problem, variant, n_eval):
    b = _best(problem, variant)
    return b[:, min(n_eval, b.shape[1]) - 1] - C.GLOBAL_MIN[problem]


def regret_at_rounds(problem, variant, r):
    _, q = split_variant(variant)
    b = _best(problem, variant)
    idx = min(C.N_START[problem] + r * q, b.shape[1]) - 1
    return b[:, idx] - C.GLOBAL_MIN[problem]


def table(problem):
    """Je Variante: Median und 90-%-Quantil des Regrets nach Budget, Median nach 3, 5, 9 Runden, mittlere Streuung der Punkte einer Runde."""
    out = {}
    for v in study()["runs"][problem]:
        fin = regret_at_evaluations(problem, v, C.TOTAL[problem])
        out[v] = {
            "final": float(np.median(fin)), "q90": float(np.quantile(fin, 0.9)), "rounds_total": rounds_in_budget(problem, v),
            "at_rounds": {r: float(np.median(regret_at_rounds(problem, v, r))) for r in C.ROUND_CHECKPOINTS},
            "spread": float(np.median(study()["runs"][problem][v]["spread"])) if study()["runs"][problem][v]["spread"] else None,
        }
    return out


def curves(problem, variant, by="evaluations"):
    """Median und 25-/75-%-Quantil des Regrets über Auswertungen oder Runden (alle festen Läufe)."""
    b = _best(problem, variant) - C.GLOBAL_MIN[problem]
    if by == "rounds":
        _, q = split_variant(variant)
        n0 = C.N_START[problem]
        idx = [min(n0 + r * q, b.shape[1]) - 1 for r in range(rounds_in_budget(problem, variant) + 1)]
        b = b[:, idx]
    else:
        b = b[:, C.N_START[problem] - 1:]
    return {"median": np.median(b, axis=0), "q25": np.quantile(b, 0.25, axis=0), "q75": np.quantile(b, 0.75, axis=0)}


def paired_stats(problem, a, b, by="rounds", at=None, tol=1e-9):
    """Gepaarter Vergleich der Varianten a und b (gleiche Seeds): besser, schlechter, gleich, Median der Differenz (a minus b; negativ = a besser), p (Vorzeichentest).
    by='rounds': bei `at` Runden (Standard: Zahl der Runden von a); by='evaluations': bei `at` Auswertungen (Standard: Budget)."""
    from scipy.stats import binomtest
    if by == "rounds":
        r = at if at is not None else rounds_in_budget(problem, a)
        x, y = regret_at_rounds(problem, a, r), regret_at_rounds(problem, b, r)
    else:
        n = at or C.TOTAL[problem]
        x, y = regret_at_evaluations(problem, a, n), regret_at_evaluations(problem, b, n)
    diff = x - y
    better, worse = int((diff < -tol).sum()), int((diff > tol).sum())
    p = float(binomtest(better, better + worse, 0.5).pvalue) if better + worse else 1.0
    return {"better": better, "worse": worse, "equal": int(len(diff) - better - worse), "median_diff": float(np.median(diff)), "p": p, "n": int(len(diff))}


def verdict_of(stats, alpha=0.05):
    if stats["p"] < alpha:
        return "besser" if stats["better"] > stats["worse"] else "schlechter"
    return "kein Unterschied nachweisbar"
