# Parallele Auswertungen: Runden gegen Auswertungen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-batch-bo-demo.streamlit.app/)** -->

Sechstes Stück der **Bayessche-Optimierung-Linie** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es setzt [bayes-opt-demo](https://github.com/sebastian-hanisch/bayes-opt-demo) fort und fragt, was passiert, wenn mehrere Auswertungen **gleichzeitig** laufen. Im Unterschied zu den
Fall-Demos zeigt die Demo ein einziges Verfahren an einem wachsenden Beispiel.

## Warum dieses Problem

Eine teure Auswertung läuft oft auf mehreren Rechnern gleichzeitig. Die Bayessche Optimierung schlägt dann q Punkte auf einmal vor, ohne die Ergebnisse der anderen zu kennen.
Zwei Zahlen zählen: die **Auswertungen** (Aufwand) und die **Runden** (Wanduhrzeit). Die Kernfrage: **Was gewinnt man mit q Punkten je Runde, und was kostet es?**

## Modell

- **Aufgabe:** Minimierung auf dem Einheitswürfel, rauschfrei. Branin (2 Eingaben, 6 Startpunkte, 30 Auswertungen), Hartmann6 (6 Eingaben, 12 Startpunkte, 48 Auswertungen) und
  Ackley (6 Eingaben, alle wirksam, 12 Startpunkte, 48 Auswertungen).
- **Ersatzmodell:** Gauß-Prozess mit Matérn-5/2-Kern, je Eingabe ein Längenmaß, Hyperparameter per Randwahrscheinlichkeit (L-BFGS-B). Akquisition: erwartete Verbesserung über
  etwa 1 950 Kandidaten.
- **Batch-Regeln:** *Kriging Believer* (Scheinbeobachtung = Mittelwert des Modells), *Constant Liar* (Scheinbeobachtung = bester Wert), *Thompson-Batch* (je Punkt eine
  Stichprobe aus dem Modell), *die q besten EI-Punkte* ohne Abstimmung. Dazu der sequentielle Lauf (q = 1). Batchgrößen q = 2, 4, 8.
- **Zwei Maßstäbe:** gleiche *Auswertungen* (gleiches Budget) und gleiche *Runden* (gleiche Wanduhrzeit, jede Runde dauert gleich lang, q Rechner stehen bereit).
- **Live-Teil:** ein Lauf, Batch gegen sequentiell, mit denselben Startpunkten. Die Studie ist vorgerechnet (`precomputed_study.json`, 780 Läufe, `generate_study.py`).
- Mathematik im 📐-Expander der App.

## Methodik

Je Zelle (Funktion, Variante) **20 feste Läufe** (Seeds 800000 bis 800019); alle Varianten eines Seeds beginnen mit **denselben** Startpunkten (gepaarter Vergleich). Verglichen wird
der Regret (bester Wert minus Minimum). Unterschiede werden mit dem Vorzeichentest geprüft (`scipy.stats.binomtest`); „kein Unterschied nachweisbar“ heißt nicht „kein Unterschied“.

## Befunde (gemessen, keine Behauptungen)

Median des Regrets nach dem Budget (Zahl der Runden in Klammern):

| Funktion | sequentiell | Kriging Believer q = 2 / 4 / 8 | Constant Liar q = 2 / 4 / 8 | Thompson q = 2 / 4 / 8 | die q besten Punkte q = 2 / 4 / 8 |
|---|---|---|---|---|---|
| Branin (30) | 0,005 (24) | 0,002 / 0,005 / 0,016 (12 / 6 / 3) | 0,007 / 0,009 / 0,031 | 0,028 / 0,028 / 0,195 | 0,008 / 0,063 / 0,530 |
| Hartmann6 (48) | 0,045 (36) | 0,089 / 0,184 / 0,243 (18 / 9 / 5) | 0,060 / 0,131 / 0,224 | 0,150 / 0,283 / 0,372 | 0,082 / 0,128 / 0,192 |
| Ackley, 6 Eingaben (48) | 3,65 (36) | 3,36 / 3,84 / 4,39 (18 / 9 / 5) | 3,56 / 3,90 / 3,92 | 4,03 / 4,70 / 4,73 | 3,65 / 3,44 / 4,09 |

| Frage | Befund |
|---|---|
| Kostet der Batch Güte je Auswertung? | **Ja, auf Hartmann6 in allen zwölf Varianten** (Kriging Believer q = 4: 0,18 gegen 0,045, 0 : 20 Läufe, p < 0,001). Auf Branin kostet es bei q = 2 und q = 4 mit Kriging Believer nichts Nachweisbares, bei q = 8 schon. Auf Ackley verlieren nur vier der zwölf Varianten nachweisbar, weil schon der sequentielle Lauf kaum lernt. |
| Gewinnt der Batch je Runde? | **Ja, in allen 36 Zellen** (3 Funktionen x 12 Varianten, p < 0,05, meist 19 : 1 oder 20 : 0). Hartmann6 nach 9 Runden: Kriging Believer q = 4 Regret 0,18 gegen sequentiell 1,22. |
| Welche Regel? | Der **Thompson-Batch ist die schwächste**: Kriging Believer ist in 7 von 9 Zellen besser (p < 0,05). Zwischen Kriging Believer, Constant Liar und den q besten Punkten trennt der Test auf den Funktionen mit sechs Eingaben nicht. |
| Schadet fehlende Abstimmung? | Nur auf **Branin**: Kriging Believer ist dort bei q = 4 (17 : 3) und q = 8 (18 : 2) besser als die q besten Punkte. Der mittlere Abstand der Punkte einer Runde ist ohne Abstimmung deutlich kleiner (Branin q = 2: 0,014 gegen 0,34). |
| Wie groß darf q sein? | Je größer q, desto weniger Runden und desto schlechter das Ergebnis nach dem Budget; auf Hartmann6 Kriging Believer 0,089 / 0,184 / 0,243 bei 18 / 9 / 5 Runden. |

Jede Zahl ist in `tests/test_claims.py` nachgerechnet.

## Ehrliche Grenzen

- **Jede Runde dauert gleich lang** und q Rechner stehen bereit; dauern Auswertungen unterschiedlich lang, fressen Wartezeiten die Ersparnis.
- **Rechenzeit des Modells ist nicht eingerechnet** (je Runde ein Modell und die Wahl von q Punkten); gegenüber einer teuren Auswertung ist sie klein, bei billiger Auswertung nicht.
- **Drei glatte, rauschfreie Testfunktionen** mit zwei und sechs Eingaben. Ackley zeigt vor allem, dass bei einer Funktion, die BO in 48 Auswertungen kaum löst, Parallelität nichts mehr
  verschlechtert, weil nichts mehr zu verlieren ist.
- **20 Läufe je Zelle:** der Test trennt nur klare Unterschiede; die Reihenfolge der drei Regeln Kriging Believer, Constant Liar und die q besten Punkte ist auf Hartmann6 und Ackley
  **nicht gesichert**.
- **Eigene, einfache Implementierung** (Kandidatenmenge statt Gradientenverfahren, Thompson-Stichproben an 600 Kandidaten). q-EI per Monte-Carlo (BoTorch) und lokale Bestrafung sind
  nicht verglichen; Zahlen anderer Bibliotheken können abweichen.
- **Ein Startdesign je Seed** (gleichverteilt, nicht Sobol): bei 6 oder 12 Startpunkten streut das Ergebnis mit dem Startdesign; deshalb der gepaarte Vergleich.

## Verifikation

- **Kern:** Kern, erwartete Verbesserung, Scheinbeobachtung und Stichprobe gegen Handrechnung auf Mini-Instanzen; Interpolation rauschfreier Daten; Batches liegen im Würfel; der
  Kriging-Believer-Batch ist breiter gestreut als der naive.
- **Orakel:** Gauß-Prozess-Vorhersage gegen scikit-learn bei festen Hyperparametern; Kriging-Believer-Aktualisierung gegen die Handformel der bedingten Varianz; erwartete Verbesserung
  gegen numerische Integration; Vorzeichentest gegen die Binomialformel; Studie auf Vollständigkeit, Monotonie und gemeinsames Startdesign geprüft.
- **Aussagen:** `tests/test_claims.py` rechnet jede README-Zahl über dieselben Auswertungsfunktionen wie die App nach.
- **Presets:** vollständig, Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler und jede Auswahl an beiden Enden) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `bbo_algorithm.py` | Kern: Kern, Gauß-Prozess, erwartete Verbesserung, Batch-Regeln, Suchverfahren |
| `bbo_scenario.py` | Branin, Hartmann6, Ackley |
| `bbo_evaluation.py` | Läufe, vorgerechnete Studie, Verläufe über Auswertungen und Runden, gepaarte Statistik |
| `bbo_constants.py` | Funktionen, Verfahren, Studie, Regler-Grenzen, Farben, Presets |
| `bbo_presets.py` | Permalink (Begrenzen/Einrasten), Presets, Zufalls-Seed-Button |
| `bbo_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `bbo_pdf_export.py` | PDF-Auswertung des Live-Laufs |
| `generate_study.py` | erzeugt `precomputed_study.json` |
| `precomputed_study.json` | 780 Läufe der Studie |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scikit-learn |
| `tests/` | Handrechnung, Orakel, Aussagen, Presets, AppTest-Smoke-Test, Footer |

## Bewusst nicht umgesetzt

- q-EI per Monte-Carlo, lokale Bestrafung, Batch mit Rauschen: eigene Verfahren bzw. Messungen.
- Asynchrone Auswertungen (ein Rechner wird frei, sofort neuer Punkt): andere Zeitrechnung als gleich lange Runden.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
