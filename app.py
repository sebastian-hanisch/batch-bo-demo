"""Parallele Auswertungen: Runden gegen Auswertungen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Linie Bayessche Optimierung. Eine einzelne Frage: Was gewinnt man, wenn q Auswertungen gleichzeitig laufen, und was kostet es?
Live: Branin, Hartmann6 oder Ackley, ein Batch gegen den sequentiellen Lauf; Studie: 20 feste Läufe je Variante.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import bbo_constants as C
import bbo_evaluation as ev
import bbo_scenario as S
from bbo_evaluation import Settings
from bbo_pdf_export import build_pdf
from bbo_presets import apply_preset, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params, bounds
from bbo_visualization import (
    build_points_chart,
    build_regret_chart,
    build_study_chart,
    build_tradeoff_chart,
    variant_label,
)

st.set_page_config(page_title="Parallele Auswertungen – Sebastian Hanisch", layout="wide")


def de(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


@st.cache_data(show_spinner=False)
def _live(settings):
    seq = ev.run_one(settings.problem, "kb", 1, settings.seed)
    bat = ev.run_one(settings.problem, settings.method, settings.q, settings.seed)
    return seq, bat


st.title("⏱️ Parallele Auswertungen: Runden gegen Auswertungen")
st.markdown(
    """
Eine teure Auswertung (eine Simulation, ein Versuch) läuft oft auf mehreren Rechnern oder Anlagen gleichzeitig. Die Bayessche Optimierung schlägt dann nicht einen, sondern
**q Punkte auf einmal** vor, ohne die Ergebnisse der anderen zu kennen. Zwei Zahlen zählen: die **Auswertungen** (Aufwand, Geld) und die **Runden** (Wanduhrzeit). Die Demo misst
beides an drei Testfunktionen und vergleicht vier Regeln für die Wahl des Batches mit dem sequentiellen Lauf. Sie ist das sechste Stück der Linie **Bayessche Optimierung**; im
Unterschied zu den Fall-Demos zeigt sie ein einziges Verfahren an einem wachsenden Beispiel.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Was Parallelität ändert", expanded=True):
    st.markdown(
        """
1. **Das Modell sieht die Ergebnisse der laufenden Runde nicht.** Bei q Punkten je Runde fehlen für q − 1 Punkte die Rückmeldungen, die sequentiell schon vorlägen.
2. **Die Punkte müssen sich unterscheiden.** Wählt man die q besten Punkte der erwarteten Verbesserung ohne Abstimmung, liegen sie dicht beieinander und bringen wenig Neues.
3. **Vier Regeln dagegen:** *Kriging Believer* (der Mittelwert des Modells als Scheinergebnis), *Constant Liar* (der bisher beste Wert als „Lüge“), *Thompson-Batch*
   (je Punkt eine Modellstichprobe) und als Gegenprobe die **q besten Punkte** ohne Abstimmung.
4. **Gleiches Startdesign:** Alle Varianten eines Seeds beginnen mit denselben Startpunkten, damit der Vergleich gepaart ist.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Der Live-Lauf**")
    problem = st.radio("Funktion", C.PROBLEMS, format_func=lambda p: C.PROBLEM_LABELS[p], key="problem_radio",
                       help="Branin hat zwei Eingaben und lässt sich als Punktwolke zeigen; Hartmann6 und Ackley haben sechs.")
    q = st.select_slider("Punkte je Runde (q)", C.QS, key="q_select", help="Zahl der gleichzeitig laufenden Auswertungen. Der sequentielle Lauf (q = 1) ist immer die Vergleichslinie.")
    method = st.radio("Regel für den Batch", C.METHODS, format_func=lambda m: C.METHOD_LABELS[m], key="method_radio", help="Wie die q Punkte einer Runde gewählt werden.")
    st.caption(C.METHOD_HELP[method])
    seed = st.number_input("Zufalls-Seed des Laufs", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"problem_radio": problem, "q_select": int(q), "method_radio": method, "seed_input": int(seed)})

settings = Settings(problem=problem, q=int(q), method=method, seed=int(seed))
seq, bat = _live(settings)
opt = C.GLOBAL_MIN[problem]
n0, total = C.N_START[problem], C.TOTAL[problem]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"{C.PROBLEM_SHORT[problem]}: {n0} Startpunkte, dann bis zu {total} Auswertungen insgesamt. Regret = bester gefundener Wert minus Minimum ({de(opt, 5)}); 0 wäre das Optimum.")
c1, c2, c3 = st.columns(3)
c1.metric("Sequentiell: Regret", de(seq.best_so_far[-1] - opt, 3), delta=f"{seq.rounds} Runden", delta_color="off", help="Eine Auswertung je Runde.")
c2.metric(f"Batch (q = {bat.variant.split('_q')[1]}): Regret", de(bat.best_so_far[-1] - opt, 3), delta=f"{bat.rounds} Runden", delta_color="off",
          help=f"{variant_label(bat.variant)}; die letzte Runde darf kleiner sein.")
c3.metric("Wanduhrzeit gespart", f"{de(100 * (1 - bat.rounds / seq.rounds), 0)} %", help="Runden des Batches gegenüber den sequentiellen Runden.")

runs = {"seq": (seq.best_so_far, seq.round_of, n0), bat.variant: (bat.best_so_far, bat.round_of, n0)}
left, right = st.columns(2)
with left:
    st.markdown("**Über die Auswertungen** (Aufwand)")
    st.plotly_chart(build_regret_chart(runs, opt, "evaluations"), width="stretch", key="live_eval")
with right:
    st.markdown("**Über die Runden** (Wanduhrzeit)")
    st.plotly_chart(build_regret_chart(runs, opt, "rounds"), width="stretch", key="live_rounds")
sp = sum(bat.spread) / len(bat.spread) if bat.spread else 0.0
st.caption(f"Mittlerer Abstand der Punkte einer Runde (auf die Würfeldiagonale bezogen): **{de(sp, 2)}**. Je kleiner, desto dichter liegen die Punkte beieinander.")
if C.DIM[problem] == 2:
    st.markdown("**Wo der Batch hinschaut** – Punkte des Batch-Laufs, Farbe = Runde")
    st.plotly_chart(build_points_chart(bat, problem, S.objective(problem)), width="stretch", key="live_points")
st.download_button("📄 Auswertung als PDF", data=build_pdf(settings, seq, bat, opt), file_name="parallele_auswertungen_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Lohnt sich der Batch?")
tb = ev.table("hartmann")
st.info(
    f"Auf Hartmann6 (20 feste Läufe, Kriging Believer, 48 Auswertungen) liegt der Regret mit q = 4 bei {de(tb['kb_q4']['final'], 2)} nach {tb['kb_q4']['rounds_total']} Runden, "
    f"sequentiell bei {de(tb['seq']['final'], 2)} nach {tb['seq']['rounds_total']} Runden: **je Auswertung verliert der Batch, je Runde gewinnt er**. Der Ausschlag liegt bei der Frage, "
    "was knapp ist: Rechner und Geld (dann sequentiell) oder Zeit (dann parallel)."
)
st.caption("Ein einzelner Lauf zeigt nur, was bei diesem Seed geschah; die Experimente mitteln über 20 feste Läufe je Variante.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

sp_sel = st.radio("Funktion der Studie", C.PROBLEMS, format_func=lambda p: C.PROBLEM_LABELS[p], key="study_problem", horizontal=True)
T = ev.table(sp_sel)
n_eval = C.TOTAL[sp_sel]

st.subheader("🔬 Experiment 1: Je Auswertung verliert der Batch, je Runde gewinnt er")
rows = []
for m in C.METHODS:
    for qq in C.QS:
        v = f"{m}_q{qq}"
        a = ev.paired_stats(sp_sel, v, "seq", by="evaluations")
        b = ev.paired_stats(sp_sel, v, "seq", by="rounds")
        rows.append(f"| {variant_label(v)} | {T[v]['rounds_total']} | {de(T[v]['final'], 3)} | {a['better']} : {a['worse']} | {ev.verdict_of(a)} | {b['better']} : {b['worse']} | {ev.verdict_of(b)} |")
st.markdown(
    f"**{C.PROBLEM_SHORT[sp_sel]}, {n_eval} Auswertungen, 20 feste Läufe:** sequentiell Regret {de(T['seq']['final'], 3)} nach {T['seq']['rounds_total']} Runden. Gepaarter Vergleich "
    "mit dem sequentiellen Lauf (gleiche Seeds, besser : schlechter, Vorzeichentest p < 0,05):\n\n"
    "| Variante | Runden | Regret (Median) | Gleiche Auswertungen | Urteil | Gleiche Runden | Urteil |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows)
)
st.info("**Gleiche Auswertungen:** gleiches Budget, der sequentielle Lauf hat mehr Gelegenheiten, aus Ergebnissen zu lernen. **Gleiche Runden:** gleiche Wanduhrzeit, der Batch hat q mal so "
        "viele Auswertungen. Beides sind faire Vergleiche, sie beantworten verschiedene Fragen.")

st.subheader("🔬 Experiment 2: Welche Regel?")
qsel = st.select_slider("Punkte je Runde (Studie)", C.QS, value=4, key="study_q")
by = st.radio("Achse", ("rounds", "evaluations"), format_func=lambda b: "Runden" if b == "rounds" else "Auswertungen", key="study_by", horizontal=True)
curves = {"seq": ev.curves(sp_sel, "seq", by)}
for m in C.METHODS:
    curves[f"{m}_q{qsel}"] = ev.curves(sp_sel, f"{m}_q{qsel}", by)
st.plotly_chart(build_study_chart(curves, by), width="stretch", key=f"study_curves_{sp_sel}_{qsel}_{by}")
kb_topq = ev.paired_stats(sp_sel, f"kb_q{qsel}", f"topq_q{qsel}", by="evaluations")
kb_ts = ev.paired_stats(sp_sel, f"kb_q{qsel}", f"ts_q{qsel}", by="evaluations")
cl_kb = ev.paired_stats(sp_sel, f"cl_q{qsel}", f"kb_q{qsel}", by="evaluations")
st.markdown(
    f"**{C.PROBLEM_SHORT[sp_sel]}, q = {qsel}, nach {n_eval} Auswertungen** (gepaart, besser : schlechter):\n\n"
    f"- Kriging Believer gegen die q besten Punkte: {kb_topq['better']} : {kb_topq['worse']}, {ev.verdict_of(kb_topq)} (p = {de(kb_topq['p'], 3)})\n"
    f"- Kriging Believer gegen Thompson-Batch: {kb_ts['better']} : {kb_ts['worse']}, {ev.verdict_of(kb_ts)} (p = {de(kb_ts['p'], 3)})\n"
    f"- Constant Liar gegen Kriging Believer: {cl_kb['better']} : {cl_kb['worse']}, {ev.verdict_of(cl_kb)} (p = {de(cl_kb['p'], 3)})"
)
st.info("Der Thompson-Batch ist in den Messungen die schwächste Regel. Zwischen Kriging Believer, Constant Liar und den q besten Punkten trennt der Vorzeichentest bei 20 Läufen oft nicht.")

st.subheader("🔬 Experiment 3: Wie weit liegen die Punkte eines Batches auseinander?")
srows = []
for m in C.METHODS:
    srows.append(f"| {C.METHOD_LABELS[m]} | " + " | ".join(de(T[f"{m}_q{qq}"]["spread"], 2) for qq in C.QS) + " |")
st.markdown("Mittlerer paarweiser Abstand der Punkte einer Runde, auf die Würfeldiagonale bezogen (Median über alle Runden und Läufe):\n\n| Regel | q = 2 | q = 4 | q = 8 |\n|---|---|---|---|\n" + "\n".join(srows))
st.info("Ohne Abstimmung liegen die Punkte um ein Vielfaches dichter. Ob das schadet, hängt von der Funktion ab: auf Branin verliert die naive Regel deutlich, auf den Funktionen mit "
        "sechs Eingaben ist die Reihenfolge der Regeln bei 20 Läufen oft nicht gesichert.")

st.subheader("🔬 Experiment 4: Wie groß darf der Batch sein?")
trows = [(v, T[v]["rounds_total"], T[v]["final"]) for v in C.VARIANTS if v == "seq" or v.startswith("kb_")]
st.plotly_chart(build_tradeoff_chart(trows), width="stretch", key=f"tradeoff_{sp_sel}")
kb8 = T["kb_q8"]
st.info(f"Je größer q, desto weniger Runden und desto schlechter das Ergebnis nach dem Budget: auf {C.PROBLEM_SHORT[sp_sel]} {T['seq']['rounds_total']} Runden mit Regret {de(T['seq']['final'], 3)} "
        f"gegen {kb8['rounds_total']} Runden mit {de(kb8['final'], 3)} bei q = 8 (Kriging Believer). Das Verhältnis aus gesparter Zeit und verlorener Güte ist die eigentliche Entscheidung.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Jede Auswertung dauert gleich lang** | In der Studie zählt jede Runde gleich. Dauern die Auswertungen unterschiedlich, warten schnelle Rechner auf langsame, oder man startet asynchron neu. | Asynchrone Verfahren |
| **q Rechner stehen bereit** | Die Zeitersparnis gilt nur, wenn q Auswertungen wirklich gleichzeitig laufen; Warteschlangen vor den Rechnern fressen sie. | Kapazitätsplanung |
| **Die Funktion ist rauschfrei** | Verrauschte Auswertungen und Batches überlagern sich; die Studie zeigt nur den rauschfreien Fall. | Verrauschte Auswertungen |
| **Zielfunktion glatt, kleine Dimension** | Drei Testfunktionen mit zwei und sechs Eingaben; in höheren Dimensionen sind die Hyperparameter-Anpassung und die Maximierung der Akquisition schwieriger. | Hohe Dimensionen |
| **20 Läufe je Zelle** | Die Zellen trennen nur klare Unterschiede; „kein Unterschied nachweisbar“ heißt nicht „kein Unterschied“. | Mehr Läufe |
| **Eigene, einfache Implementierung** | Kandidatenmenge statt Gradientenverfahren, ein Modell je Runde. Zahlen anderer Bibliotheken (BoTorch, q-EI) können abweichen. | Vergleich mit BoTorch |
    """
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Ersatzmodell.** Gauß-Prozess mit Matérn-5/2-Kern, je Eingabe ein Längenmaß $\ell_i$:

$$k(x, x') = \sigma_f^2\Bigl(1 + \sqrt5\, r + \tfrac53 r^2\Bigr)e^{-\sqrt5\, r}, \qquad r = \sqrt{\textstyle\sum_{i=1}^{d} (x_i - x'_i)^2/\ell_i^2}$$

**Erwartete Verbesserung** (Minimierung, $z = (f^\* - \mu)/s$):

$$\mathrm{EI}(x) = (f^\* - \mu(x))\,\Phi(z) + s(x)\,\varphi(z)$$

**Kriging Believer.** Nach der Wahl von $x_j$ wird $(x_j, \mu(x_j))$ als Beobachtung ergänzt, die Hyperparameter bleiben, und der nächste Punkt maximiert EI unter dem erweiterten
Modell. **Constant Liar:** statt $\mu(x_j)$ wird der bisher beste Wert $f^\*$ ergänzt.

**Thompson-Batch.** Für $j = 1, \dots, q$ eine gemeinsame Stichprobe $\tilde f_j \sim \mathcal{GP}$ an 600 Kandidaten, dann

$$x_j = \arg\min_{x \in \text{Kandidaten}} \tilde f_j(x)$$

**Die q besten Punkte.** $x_1, \dots, x_q$ sind die q Kandidaten mit größter EI, ohne Abstimmung.

**Zwei Maßstäbe.** Nach $R$ Runden hat der Batch $n_0 + qR$ Auswertungen, der sequentielle Lauf $n_0 + R$; der Regret ist der beste Wert minus das Minimum.

**Literatur.** D. R. Jones, M. Schonlau, W. J. Welch (1998), *Efficient Global Optimization of Expensive Black-Box Functions*, J. Global Optimization 13 (EI);
D. Ginsbourger, R. Le Riche, L. Carraro (2010), *Kriging Is Well-Suited to Parallelize Optimization* (Kriging Believer, Constant Liar); J. Snoek, H. Larochelle, R. P. Adams (2012),
*Practical Bayesian Optimization of Machine Learning Algorithms*, NeurIPS; K. Kandasamy u. a. (2018), *Parallelised Bayesian Optimisation via Thompson Sampling*, AISTATS.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
