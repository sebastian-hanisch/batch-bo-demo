"""Vorgerechnete Studie: sequentiell und vier Batch-Regeln mit q = 2, 4, 8 auf Branin, Hartmann6 und Ackley (6 Eingaben), je 20 feste Läufe.
Wenige Minuten auf 14 Kernen.

    python generate_study.py            # schreibt precomputed_study.json
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import bbo_constants as C
import bbo_evaluation as ev


def job(args):
    problem, variant, seed = args
    m, q = ev.split_variant(variant)
    r = ev.run_one(problem, m if m != "seq" else "kb", q, seed)
    return problem, variant, seed, {"best": r.best_so_far, "spread": r.spread, "seconds": r.seconds, "rounds": r.rounds}


def main(workers=14):
    t0 = time.time()
    jobs = [(p, v, s) for p in C.PROBLEMS for v in C.VARIANTS for s in C.SWEEP_SEEDS]
    jobs.sort(key=lambda j: (j[1] != "seq", j[0] != "hartmann"))                 # die langsamen (sequentiell, Hartmann) zuerst
    runs = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for problem, variant, seed, rec in ex.map(job, jobs, chunksize=1):
            slot = runs.setdefault(problem, {}).setdefault(variant, {"best": [], "spread": [], "seconds": [], "seeds": [], "rounds": []})
            slot["best"].append(rec["best"])
            slot["spread"].extend(rec["spread"])
            slot["seconds"].append(rec["seconds"])
            slot["seeds"].append(seed)
            slot["rounds"].append(rec["rounds"])
    out = {"runs": runs, "meta": {"seeds": len(C.SWEEP_SEEDS), "jobs": len(jobs), "seconds": round(time.time() - t0)}}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_study.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("fertig", len(jobs), "Läufe in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
