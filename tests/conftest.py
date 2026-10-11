"""Gemeinsame Fixtures: eine kleine Beobachtungsmenge, die von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """Fünf Beobachtungen einer Funktion einer Eingabe: f(x) = (x − 0,5)², Minimum bei 0,5."""
    X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    return X, ((X[:, 0] - 0.5) ** 2)
