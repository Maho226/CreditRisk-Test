"""Tests de non-régression (Partie V) : contrat, prédictions de référence, métriques, sauvegarde/rechargement."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from conftest import CHAMPS_REPONSE, REFERENCE

from app import training
from app.non_regression import (
    REFERENCE as CHEMIN_REFERENCE,
)
from app.non_regression import (
    TOLERANCE_PROBA_CODE,
    TOLERANCE_PROBA_RECHARGEMENT,
    comparer,
    decision_metriques,
    lot_de_controle,
    metriques_validation,
    probabilites,
)
from app.prediction import charger_modele, sha256

MODELE = "models/model_v1.joblib"


@pytest.fixture(scope="module")
def reference():
    return json.loads(CHEMIN_REFERENCE.read_text())


@pytest.fixture(scope="module")
def paquet():
    return charger_modele(MODELE)


# ------------------------------------------------------------------ référence figée
def test_reference_figee_et_coherente(reference):
    assert reference["version_modele"] == "v1"
    assert reference["sha256_modele"] == sha256(MODELE)
    assert reference["sha256_donnees"] == sha256("data/credit_synthetique.csv")
    assert len(reference["lot_controle"]["demandes"]) == len(reference["lot_controle"]["probabilites"]) == 200


def test_lot_de_controle_reproductible(reference):
    lot = lot_de_controle()
    assert lot.to_dict(orient="records") == reference["lot_controle"]["demandes"]


# ------------------------------------------------------------------ CT27 : sorties du pipeline identiques
def test_probabilites_du_lot_de_controle_inchangees(reference, paquet):
    lot = pd.DataFrame(reference["lot_controle"]["demandes"])
    ecarts = np.abs(probabilites(paquet, lot) - np.array(reference["lot_controle"]["probabilites"]))
    assert ecarts.max() <= TOLERANCE_PROBA_CODE


def test_probabilites_via_l_api_inchangees(client, reference):
    """Même contrôle à travers toute la chaîne : validation, prétraitement, modèle, API."""
    demandes = reference["lot_controle"]["demandes"][:30]
    for demande, attendue in zip(demandes, reference["lot_controle"]["probabilites"][:30], strict=True):
        reponse = client.post("/predict", json=demande)
        assert reponse.status_code == 200
        assert abs(reponse.json()["probabilite_defaut"] - attendue) <= TOLERANCE_PROBA_CODE


# ------------------------------------------------------------------ sorties conformes au contrat
def test_sorties_conformes_au_contrat_sur_tout_le_lot(client, reference):
    for demande in reference["lot_controle"]["demandes"]:
        resultat = client.post("/predict", json=demande).json()
        assert set(resultat) == CHAMPS_REPONSE
        assert math.isfinite(resultat["probabilite_defaut"]) and 0 <= resultat["probabilite_defaut"] <= 1


# ------------------------------------------------------------------ CT23 : reproductibilité
def test_meme_demande_meme_probabilite(client):
    resultats = {client.post("/predict", json=REFERENCE).json()["probabilite_defaut"] for _ in range(10)}
    assert len(resultats) == 1


# ------------------------------------------------------------------ CT29 : sauvegarde et rechargement
def test_sauvegarde_puis_rechargement(tmp_path, reference, paquet):
    chemin = training.sauvegarder(
        {k: paquet[k] for k in ("pipeline", "seuil", "variables", "metadonnees")}, tmp_path / "copie.joblib"
    )
    recharge = charger_modele(chemin)  # vérifie aussi l'empreinte enregistrée dans copie.json
    lot = pd.DataFrame(reference["lot_controle"]["demandes"])
    assert np.abs(probabilites(recharge, lot) - probabilites(paquet, lot)).max() <= TOLERANCE_PROBA_RECHARGEMENT
    assert recharge["seuil"] == paquet["seuil"]


# ------------------------------------------------------------------ CT28 : métriques vs référence
def test_metriques_du_modele_actuel_egales_a_la_reference(reference, paquet):
    actuelles = metriques_validation(paquet)
    for nom in ("rappel", "precision", "f1", "roc_auc"):
        assert actuelles[nom] == pytest.approx(reference["metriques_validation"][nom], abs=1e-12)


def test_comparaison_du_modele_de_reference_avec_lui_meme():
    resultat = comparer(MODELE)
    assert resultat["decision"] == "accepter"
    # pas d'égalité stricte : les calculs flottants peuvent différer au 16e chiffre d'une machine à l'autre
    assert resultat["ecart_probabilite_max"] <= 1e-12
    assert resultat["decisions_changees"] == 0


def test_reentrainement_reproductible(reference):
    """Réentraîner v1 avec la même graine redonne le même modèle (mêmes probabilités, même seuil)."""
    paquet = training.entrainer("v1")
    lot = pd.DataFrame(reference["lot_controle"]["demandes"])
    assert paquet["seuil"] == reference["seuil"]
    assert np.abs(probabilites(paquet, lot) - np.array(reference["lot_controle"]["probabilites"])).max() <= 1e-9


# ------------------------------------------------------------------ détection d'une baisse excessive
REF = {"rappel": 0.75, "f1": 0.52, "roc_auc": 0.78}


@pytest.mark.parametrize(
    "candidat, attendu",
    [
        ({"rappel": 0.75, "f1": 0.52, "roc_auc": 0.78}, "accepter"),
        ({"rappel": 0.80, "f1": 0.55, "roc_auc": 0.80}, "accepter"),  # amélioration
        ({"rappel": 0.73, "f1": 0.50, "roc_auc": 0.77}, "accepter"),  # baisses dans les tolérances
        ({"rappel": 0.71, "f1": 0.52, "roc_auc": 0.78}, "investiguer"),  # rappel −0,04
        ({"rappel": 0.75, "f1": 0.47, "roc_auc": 0.78}, "investiguer"),  # F1 −0,05
        ({"rappel": 0.68, "f1": 0.52, "roc_auc": 0.78}, "rejeter"),  # rappel −0,07
        ({"rappel": 0.75, "f1": 0.40, "roc_auc": 0.78}, "rejeter"),  # F1 −0,12
        ({"rappel": 0.75, "f1": 0.52, "roc_auc": 0.73}, "rejeter"),  # AUC −0,05
    ],
)
def test_detection_d_une_baisse_du_rappel_ou_du_f1(candidat, attendu):
    assert decision_metriques(REF, candidat)["decision"] == attendu


def test_modele_degrade_rejete(tmp_path, paquet):
    """Un modèle volontairement dégradé (seuil relevé à 0,90) est rejeté par la comparaison."""
    degrade = {k: paquet[k] for k in ("pipeline", "variables", "metadonnees")} | {"seuil": 0.90}
    degrade["metadonnees"] = {**paquet["metadonnees"], "version": "v1-degrade"}
    chemin = training.sauvegarder(degrade, tmp_path / "degrade.joblib")
    resultat = comparer(chemin)
    assert resultat["decision"] == "rejeter"
    assert resultat["details"]["rappel"]["verdict"] == "rejeter"
    assert resultat["decisions_changees"] > 0


def test_comparaison_v2_signale_la_baisse_du_f1():
    """v2 (seuil abaissé pour le rappel) : gain de rappel, baisse du F1 au-delà de la tolérance → investiguer."""
    resultat = comparer("models/model_v2.joblib")
    assert resultat["decision"] == "investiguer"
    assert resultat["details"]["f1"]["verdict"] == "investiguer"
    assert resultat["details"]["rappel"]["baisse"] < 0  # le rappel progresse
    assert resultat["ecart_probabilite_max"] <= 1e-12  # même pipeline : seules les décisions changent
    assert resultat["decisions_changees"] > 0
