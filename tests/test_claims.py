"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (vorgerechnete Studie: 3 Funktionen x 13 Varianten x 20 feste Läufe)."""

import pytest

import bbo_constants as C
import bbo_evaluation as ev

BATCH = [f"{m}_q{q}" for m in C.METHODS for q in C.QS]


def _final(p, v):
    return ev.table(p)[v]["final"]


def test_rounds_in_budget_are_ceil_of_the_remaining_evaluations_over_q():
    for p in C.PROBLEMS:
        assert ev.table(p)["seq"]["rounds_total"] == C.TOTAL[p] - C.N_START[p]
    assert [ev.table("hartmann")[f"kb_q{q}"]["rounds_total"] for q in C.QS] == [18, 9, 5]
    assert [ev.table("branin")[f"kb_q{q}"]["rounds_total"] for q in C.QS] == [12, 6, 3]


def test_hartmann_every_batch_loses_per_evaluation_and_wins_per_round():
    for v in BATCH:
        a = ev.paired_stats("hartmann", v, "seq", by="evaluations")
        b = ev.paired_stats("hartmann", v, "seq", by="rounds")
        assert a["p"] < 0.05 and a["worse"] > a["better"], v
        assert b["p"] < 0.05 and b["better"] > b["worse"], v
    assert _final("hartmann", "seq") == pytest.approx(0.045, abs=0.01)
    assert _final("hartmann", "kb_q4") == pytest.approx(0.184, abs=0.02)
    assert _final("hartmann", "kb_q8") == pytest.approx(0.243, abs=0.02)


def test_branin_small_believer_batches_lose_nothing_provable_but_big_ones_do():
    for v in ("kb_q2", "kb_q4", "cl_q2", "topq_q2"):
        assert ev.verdict_of(ev.paired_stats("branin", v, "seq", by="evaluations")) == "kein Unterschied nachweisbar", v
    for v in ("kb_q8", "topq_q4", "topq_q8", "ts_q2", "ts_q4", "ts_q8", "cl_q8"):
        assert ev.verdict_of(ev.paired_stats("branin", v, "seq", by="evaluations")) == "schlechter", v


def test_ackley_loses_little_per_evaluation_because_the_sequential_run_barely_learns():
    worse = [v for v in BATCH if ev.verdict_of(ev.paired_stats("ackley", v, "seq", by="evaluations")) == "schlechter"]
    assert sorted(worse) == sorted(["kb_q8", "cl_q8", "ts_q4", "ts_q8"])
    assert 3.0 < _final("ackley", "seq") < 4.2


def test_every_variant_wins_per_round_on_every_function():
    for p in C.PROBLEMS:
        for v in BATCH:
            s = ev.paired_stats(p, v, "seq", by="rounds")
            assert s["p"] < 0.05 and s["better"] > s["worse"], (p, v)


def test_thompson_batch_is_the_weakest_rule():
    wins = 0
    for p in C.PROBLEMS:
        for q in C.QS:
            s = ev.paired_stats(p, f"kb_q{q}", f"ts_q{q}", by="evaluations")
            wins += int(s["p"] < 0.05 and s["better"] > s["worse"])
    assert wins == 7
    assert _final("hartmann", "ts_q8") == pytest.approx(0.372, abs=0.03)


def test_naive_top_q_loses_to_believer_on_branin_only():
    for q in (4, 8):
        s = ev.paired_stats("branin", f"kb_q{q}", f"topq_q{q}", by="evaluations")
        assert s["p"] < 0.05 and s["better"] > s["worse"], q
    for p in ("hartmann", "ackley"):
        for q in C.QS:
            assert ev.paired_stats(p, f"kb_q{q}", f"topq_q{q}", by="evaluations")["p"] > 0.05, (p, q)


def test_batch_points_are_spread_wider_with_believer_than_without_coordination():
    for p in C.PROBLEMS:
        for q in C.QS:
            assert ev.table(p)[f"kb_q{q}"]["spread"] > ev.table(p)[f"topq_q{q}"]["spread"], (p, q)
    t = ev.table("branin")
    assert t["topq_q2"]["spread"] == pytest.approx(0.014, abs=0.005) and t["kb_q4"]["spread"] == pytest.approx(0.417, abs=0.03)
