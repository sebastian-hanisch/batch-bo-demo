"""PDF-Export des Live-Laufs (fpdf2, Kernschriften; Umlaute sind dort erlaubt, Gedankenstrich und Eurozeichen nicht)."""

from fpdf import FPDF

import bbo_constants as C


def _num(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def build_pdf(settings, seq, bat, optimum):
    """Einseitiger Bericht: Einstellungen, Regret und Runden des sequentiellen und des Batch-Laufs. Gibt die Bytes der PDF-Datei zurück."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Parallele Auswertungen: Runden gegen Auswertungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Sebastian Hanisch, Operations Research und Machine Learning", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for k, v in (("Funktion", C.PROBLEM_LABELS[settings.problem]), ("Punkte je Runde (q)", str(settings.q)), ("Regel", C.METHOD_LABELS[settings.method]),
                 ("Zufalls-Seed", str(settings.seed))):
        pdf.cell(60, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Ergebnis nach dem Budget", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for name, run in (("Sequentiell", seq), ("Batch", bat)):
        pdf.cell(0, 6, f"{name}: Regret {_num(run.best_so_far[-1] - optimum, 3)}, {run.rounds} Runden, {len(run.best_so_far)} Auswertungen", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Ein einzelner Lauf zeigt nur, was bei diesem Seed geschah; die Studie der Demo mittelt über 20 feste Läufe je Variante.")
    return bytes(pdf.output())
