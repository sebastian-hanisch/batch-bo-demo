"""Konstanten der Demo „Parallele Auswertungen“: Testfunktionen, Batch-Verfahren, Studie, Regler, Presets, Farben."""

# --- Testfunktionen (Einheitswürfel, Minimierung, rauschfrei) -------------------------------------------------------------------------------
PROBLEMS = ("branin", "hartmann", "ackley")
PROBLEM_LABELS = {"branin": "Branin (2 Eingaben)", "hartmann": "Hartmann6 (6 Eingaben)", "ackley": "Ackley (6 Eingaben, alle wirksam)"}
PROBLEM_SHORT = {"branin": "Branin", "hartmann": "Hartmann6", "ackley": "Ackley6"}
DIM = {"branin": 2, "hartmann": 6, "ackley": 6}
GLOBAL_MIN = {"branin": 0.397887, "hartmann": -3.32237, "ackley": 0.0}
N_START = {"branin": 6, "hartmann": 12, "ackley": 12}             # Startdesign (gleichverteilt), gleich für alle Varianten
TOTAL = {"branin": 30, "hartmann": 48, "ackley": 48}              # Auswertungen insgesamt

# --- Verfahren ----------------------------------------------------------------------------------------------------------------------------
METHODS = ("topq", "kb", "cl", "ts")
METHOD_LABELS = {
    "seq": "Sequentiell (q = 1)",
    "topq": "Die q besten EI-Punkte",
    "kb": "Kriging Believer",
    "cl": "Constant Liar (Lüge = bester Wert)",
    "ts": "Thompson-Batch",
}
METHOD_HELP = {
    "topq": "Naiv: die q Kandidaten mit der größten erwarteten Verbesserung, ohne dass ein Punkt die anderen kennt. Die Punkte liegen dicht beieinander.",
    "kb": "Der erste Punkt wird gewählt, das Modell nimmt als Scheinbeobachtung seinen eigenen Mittelwert dort an, dann folgt der nächste Punkt, und so weiter.",
    "cl": "Wie Kriging Believer, aber die Scheinbeobachtung ist der bisher beste Wert (die „Lüge“): drückt die Unsicherheit um die gewählten Punkte weg.",
    "ts": "Jeder Punkt ist das Minimum einer eigenen Stichprobe aus dem Modell (Thompson-Stichprobe): Vielfalt kommt aus dem Zufall.",
}
QS = (2, 4, 8)
LIVE_QS = QS
VARIANTS = ("seq",) + tuple(f"{m}_q{q}" for m in METHODS for q in QS)
METHOD_COLORS = {"seq": "#14233B", "topq": "#B3463A", "kb": "#2F6B65", "cl": "#7B5EA7", "ts": "#D68A2E"}
RANDOM_COLOR = "#8c8c8c"

# --- Studie -------------------------------------------------------------------------------------------------------------------------------
SWEEP_SEEDS = tuple(range(800000, 800020))
ROUND_CHECKPOINTS = (3, 5, 9)

# --- Regler -----------------------------------------------------------------------------------------------------------------------------
SEED_MAX, DEFAULT_SEED = 999999, 35

PRESETS = {
    "Standardfall (Voreinstellung)": {"problem": "branin", "q": 4, "method": "kb", "seed": DEFAULT_SEED},
    "Naiver Batch": {"problem": "branin", "q": 4, "method": "topq", "seed": DEFAULT_SEED},
    "Großer Batch": {"problem": "branin", "q": 8, "method": "kb", "seed": DEFAULT_SEED},
    "Thompson-Batch": {"problem": "branin", "q": 4, "method": "ts", "seed": DEFAULT_SEED},
    "Hartmann6": {"problem": "hartmann", "q": 4, "method": "kb", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Branin (2 Eingaben), vier Punkte je Runde nach der Kriging-Believer-Regel, verglichen mit dem sequentiellen Lauf.",
    "Naiver Batch": "Die vier besten Punkte nach erwarteter Verbesserung, ohne Abstimmung: sie liegen dicht beieinander.",
    "Großer Batch": "Acht Punkte je Runde: nur drei Runden bis zum Budget, aber jede Auswertung ist weniger wert.",
    "Thompson-Batch": "Jeder Punkt ist das Minimum einer eigenen Modellstichprobe.",
    "Hartmann6": "Sechs Eingaben statt zwei: Der Batch verliert je Auswertung deutlicher, gewinnt je Runde weiterhin.",
}

COLOR_OBS = "#2F6B65"
COLOR_BAD = "#B3463A"
COLOR_NEXT = "#D68A2E"
