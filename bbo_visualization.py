"""Plotly-Abbildungen (alle Achsen fest, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go

import bbo_constants as C


def _fixed(fig, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    return fig


def _color(variant):
    return C.METHOD_COLORS[variant.split("_q")[0]]


def build_points_chart(run, problem, f=None):
    """Nur für zwei Eingaben: ausgewertete Punkte im Einheitsquadrat, Startdesign grau, Runden farbig (Reihenfolge der Runden), mit Linien der Funktion im Hintergrund."""
    fig = go.Figure()
    if f is not None:
        g = np.linspace(0, 1, 60)
        Z = np.array([[f(np.array([u, v])) for u in g] for v in g])
        fig.add_trace(go.Contour(x=g, y=g, z=np.log1p(Z - Z.min()), showscale=False, colorscale="Greys", opacity=0.35, contours_coloring="lines", line_width=1, hoverinfo="skip"))
    X, rnd = run.X, np.array(run.round_of)
    start = rnd == 0
    fig.add_trace(go.Scatter(x=X[start, 0], y=X[start, 1], mode="markers", name="Startdesign", marker=dict(color="#8c8c8c", size=9, symbol="square")))
    later = ~start
    fig.add_trace(go.Scatter(x=X[later, 0], y=X[later, 1], mode="markers", name="Batch-Runden (Farbe = Runde)", marker=dict(color=rnd[later], colorscale="Viridis", size=10,
                  line=dict(width=1, color="white"), showscale=True, colorbar=dict(title="Runde", thickness=10)),
                  hovertemplate="Runde %{marker.color}<extra></extra>"))
    fig.update_xaxes(range=[-0.02, 1.02], title_text="Eingabe 1")
    fig.update_yaxes(range=[-0.02, 1.02], title_text="Eingabe 2", scaleanchor="x", scaleratio=1)
    return _fixed(fig, height=420)


def build_regret_chart(runs, optimum, by):
    """Regret (logarithmisch) über Auswertungen (by='evaluations') oder Runden (by='rounds'), je Variante eine Linie."""
    fig = go.Figure()
    for variant, (best, round_of, n0) in runs.items():
        best = np.maximum(np.asarray(best, float) - optimum, 1e-4)
        if by == "evaluations":
            x, y = list(range(n0, len(best) + 1)), list(best[n0 - 1:])
        else:
            rnd = np.asarray(round_of)
            last = {r: i for i, r in enumerate(rnd)}                    # letzte Auswertung je Runde
            x = sorted(last)
            y = [best[last[r]] for r in x]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=variant_label(variant), line=dict(color=_color(variant), width=2.5), line_shape="hv", marker=dict(size=5)))
    fig.update_xaxes(title_text="Auswertungen" if by == "evaluations" else "Runden (Wanduhrzeit, jede Runde dauert gleich lang)")
    fig.update_yaxes(title_text="Regret (log)", type="log")
    return _fixed(fig, height=340)


def variant_label(variant):
    m, _, q = variant.partition("_q")
    base = C.METHOD_LABELS[m]
    return base if variant == "seq" else f"{base}, q = {q}"


def build_study_chart(curves_by_variant, by):
    """Median und mittlere Hälfte der festen Läufe über Auswertungen oder Runden (Regret, logarithmisch)."""
    fig = go.Figure()
    for variant, c in curves_by_variant.items():
        n = len(c["median"])
        x = list(range(n)) if by == "rounds" else list(range(1, n + 1))
        col = _color(variant)
        lo, hi = np.maximum(c["q25"], 1e-4), np.maximum(c["q75"], 1e-4)
        fig.add_trace(go.Scatter(x=x + x[::-1], y=list(hi) + list(lo)[::-1], fill="toself", fillcolor=col, opacity=0.08, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=np.maximum(c["median"], 1e-4), mode="lines", name=variant_label(variant), line=dict(color=col, width=2.5)))
    fig.update_xaxes(title_text="Runden nach dem Startdesign" if by == "rounds" else "Auswertungen nach dem Startdesign")
    fig.update_yaxes(title_text="Regret, Median und 25 bis 75 % (log)", type="log")
    return _fixed(fig, height=400)


def build_tradeoff_chart(rows):
    """Zielkonflikt: je Variante Regret nach dem Budget (y) gegen Zahl der Runden (x); Punkte unten links sind gut."""
    fig = go.Figure()
    for variant, rounds, regret in rows:
        fig.add_trace(go.Scatter(x=[rounds], y=[max(regret, 1e-4)], mode="markers+text", text=[variant.replace("_q", " q=")], textposition="top center",
                                 marker=dict(color=_color(variant), size=13), name=variant_label(variant), showlegend=False))
    fig.update_xaxes(title_text="Runden bis zum Ende des Budgets (weniger = schneller fertig)")
    fig.update_yaxes(title_text="Regret nach dem Budget, Median (log)", type="log")
    return _fixed(fig, height=380)
