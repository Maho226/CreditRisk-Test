"""Éléments partagés par les tests de l'API, d'intégration, de robustesse et de non-régression."""

import pytest
from fastapi.testclient import TestClient

from app import main

REFERENCE = {"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000, "duree_mois": 12, "anciennete_activite": 4}
CHAMPS_REPONSE = {"probabilite_defaut", "niveau_risque", "defaut_predit", "alertes", "version_modele"}
TRACES_INTERDITES = ("Traceback", 'File "', ".py", "Exception", "Error", "joblib", "sklearn", "/content", "/home")


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def sans_modele(monkeypatch):
    """Simule un modèle non chargé (indisponible)."""
    monkeypatch.setattr(main.service, "paquet", None)


def avec(**modifs) -> dict:
    return {**REFERENCE, **modifs}


def sans_trace_technique(texte: str) -> bool:
    return not any(motif in texte for motif in TRACES_INTERDITES)
