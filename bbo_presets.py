"""Permalink (Begrenzen/Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import bbo_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None       # erlaubte Werte bei Auswahlreglern (Permalink rastet ein bzw. fällt auf den Standard zurück)


SETTING_SPECS = {
    "problem_radio": SettingSpec("problem", str, "branin", choices=C.PROBLEMS),
    "q_select": SettingSpec("q", int, 4, choices=C.QS),
    "method_radio": SettingSpec("method", str, "kb", choices=C.METHODS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "study_problem": SettingSpec("sp", str, "hartmann", choices=C.PROBLEMS),
}
PRESET_KEYS = {"problem": "problem_radio", "q": "q_select", "method": "method_radio", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def snap_to_choice(value, choices, default):
    """Erlaubter Wert: bei Zahlen der nächste, bei Zeichenketten nur ein Treffer, sonst der Standard."""
    if isinstance(default, str):
        return value if value in choices else default
    return min(choices, key=lambda c: abs(c - value))


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, begrenzt bzw. eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.lo is not None:
                value = max(spec.lo, value)
            if spec.hi is not None:
                value = min(spec.hi, value)
            if spec.choices is not None:
                value = snap_to_choice(value, spec.choices, spec.default)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
