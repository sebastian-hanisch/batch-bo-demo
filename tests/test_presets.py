"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen."""

import bbo_constants as C
import bbo_evaluation as ev
import bbo_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid():
    for preset in C.PRESETS.values():
        assert preset["problem"] in C.PROBLEMS and preset["q"] in C.QS and preset["method"] in C.METHODS
        assert 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall (Voreinstellung)"]
    assert ev.Settings(problem=p["problem"], q=p["q"], method=p["method"], seed=p["seed"]) == ev.Settings()
    for key, state_key in P.PRESET_KEYS.items():
        assert p[key] == P.SETTING_SPECS[state_key].default


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)


def test_snap_to_choice():
    assert P.snap_to_choice(5, C.QS, 4) in (4, 8)
    assert P.snap_to_choice("nope", C.METHODS, "kb") == "kb"
