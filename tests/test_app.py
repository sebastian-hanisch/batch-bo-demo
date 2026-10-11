"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_select_slider_and_radio_at_each_end_does_not_raise():
    first = _fresh_app()
    n_sliders, n_radios = len(first.select_slider), len(first.radio)
    assert n_sliders and n_radios, "keine Regler gefunden"
    for index in range(n_sliders):
        options = first.select_slider[index].options
        for pick in (0, len(options) - 1):
            at = _fresh_app()
            at.select_slider[index].set_value(at.select_slider[index].options[pick]).run(timeout=TIMEOUT)
            assert not at.exception, (index, pick, [str(e) for e in at.exception])
    for index in range(n_radios):
        options = first.radio[index].options
        for pick in (0, len(options) - 1):
            at = _fresh_app()
            at.radio[index].set_value(at.radio[index].options[pick]).run(timeout=TIMEOUT)
            assert not at.exception, (index, pick, [str(e) for e in at.exception])
