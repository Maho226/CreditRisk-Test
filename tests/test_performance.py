"""Tests de performance (Partie VI), marqués `slow` : exclus des Pull Requests, lancés manuellement.

Vérifications en processus (sans réseau), sur des demandes variées, après échauffement.
Les mesures complètes (HTTP, charge progressive, Locust) sont produites par perf/mesure_performance.py.
"""

import time

import numpy as np
import pandas as pd
import pytest

from app import main
from app.preprocessing import construire_variables

pytestmark = pytest.mark.slow

P95_API_MAX_MS = 100.0
P95_MODELE_MAX_MS = 20.0
DEBIT_MIN = 50.0


@pytest.fixture(scope="module")
def demandes():
    import json
    from pathlib import Path

    return json.loads(Path("reports/reference_v1.json").read_text())["lot_controle"]["demandes"]


def p95_ms(fonction, entrees, n=500, echauffement=50):
    durees = []
    for i in range(n + echauffement):
        debut = time.perf_counter()
        fonction(entrees[i % len(entrees)])
        if i >= echauffement:
            durees.append(time.perf_counter() - debut)
    return float(np.percentile(np.array(durees) * 1000, 95))


def test_latence_inference_du_modele(demandes):
    pipeline = main.service.paquet["pipeline"]
    X = [construire_variables(pd.DataFrame([d])) for d in demandes]
    assert p95_ms(pipeline.predict_proba, X) <= P95_MODELE_MAX_MS


def test_latence_p95_de_l_api(client, demandes):
    assert p95_ms(lambda d: client.post("/predict", json=d), demandes) <= P95_API_MAX_MS


def test_debit_de_l_api(client, demandes):
    n = 500
    debut = time.perf_counter()
    for i in range(n):
        assert client.post("/predict", json=demandes[i % len(demandes)]).status_code == 200
    assert n / (time.perf_counter() - debut) >= DEBIT_MIN


def test_taille_du_modele():
    from pathlib import Path

    assert Path("models/model_v1.joblib").stat().st_size <= 1_000_000
