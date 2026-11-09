"""Tests d'intégration (Partie III) : validation → prétraitement → modèle → API, et gestion des pannes."""

import json
import logging
import shutil

import joblib
import numpy as np
import pytest
from conftest import REFERENCE, sans_trace_technique

from app import main, prediction
from app.prediction import ModeleIndisponible, ServicePrediction, charger_modele, interpreter, sha256
from app.preprocessing import VARIABLES_MODELE, PretraitementErreur, construire_variables, demande_vers_dataframe
from app.validation import valider_demande

MODELE = "models/model_v1.joblib"


def verifier_500(reponse):
    assert reponse.status_code == 500
    assert reponse.json() == {
        "erreur": "erreur_interne",
        "message": "Une erreur interne est survenue. La demande n'a pas été traitée.",
    }
    assert sans_trace_technique(reponse.text)


# ================================================================== scénario principal
def test_scenario_principal_de_bout_en_bout(client, monkeypatch):
    """Requête valide : validation, transformation, prédiction, réponse et métadonnées cohérentes."""
    transformations = []
    origine = prediction.construire_variables

    def espion(df):
        X = origine(df)
        transformations.append(X)
        return X

    monkeypatch.setattr(prediction, "construire_variables", espion)

    # 1. requête valide
    reponse = client.post("/predict", json=REFERENCE)
    assert reponse.status_code == 200

    # 2. validation des données : la demande transmise est la demande validée
    demande = valider_demande(REFERENCE)
    assert demande.model_dump() == REFERENCE

    # 3. transformation des variables : appelée une fois, colonnes et valeurs attendues
    assert len(transformations) == 1
    X = transformations[0]
    assert list(X.columns) == VARIABLES_MODELE and X.shape == (1, 6)
    assert X.loc[0, "taux_endettement"] == pytest.approx((1_000_000 / 12) / 250_000)

    # 4. prédiction : même probabilité que le modèle appelé directement
    paquet = charger_modele(MODELE)
    attendu = paquet["pipeline"].predict_proba(construire_variables(demande_vers_dataframe(demande)))[0, 1]
    resultat = reponse.json()
    assert resultat["probabilite_defaut"] == pytest.approx(attendu, abs=1e-12)

    # 5. structure et cohérence de la réponse
    assert resultat["niveau_risque"] == interpreter(resultat["probabilite_defaut"])
    assert resultat["defaut_predit"] == (resultat["probabilite_defaut"] >= paquet["seuil"])
    assert resultat["alertes"] == []

    # 6. métadonnées cohérentes avec le modèle réellement chargé
    info = client.get("/model-info").json()
    fiche = json.loads(open("models/model_v1.json").read())
    assert info["version"] == resultat["version_modele"] == fiche["version"]
    assert info["sha256_modele"] == sha256(MODELE) == fiche["sha256_modele"]
    assert info["seuil_decision"] == paquet["seuil"]
    assert info["variables_modele"] == VARIABLES_MODELE
    assert info["sha256_donnees"] == sha256("data/credit_synthetique.csv")


def test_dimensions_du_modele_conformes_au_pretraitement():
    paquet = charger_modele(MODELE)
    assert paquet["pipeline"].n_features_in_ == len(VARIABLES_MODELE)
    assert list(paquet["pipeline"].feature_names_in_) == VARIABLES_MODELE


# ================================================================== le modèle ne peut pas être chargé
def test_fichier_modele_absent(tmp_path):
    with pytest.raises(ModeleIndisponible, match="introuvable"):
        charger_modele(tmp_path / "absent.joblib")


def test_fichier_modele_corrompu(tmp_path):
    corrompu = tmp_path / "modele.joblib"
    corrompu.write_bytes(b"ceci n'est pas un modele")
    with pytest.raises(ModeleIndisponible, match="illisible"):
        charger_modele(corrompu)


def test_fichier_modele_modifie_empreinte_differente(tmp_path):
    shutil.copy(MODELE, tmp_path / "modele.joblib")
    shutil.copy("models/model_v1.json", tmp_path / "modele.json")
    with open(tmp_path / "modele.joblib", "ab") as f:
        f.write(b"\x00")  # un octet ajouté : le fichier n'est plus celui qui a été validé
    with pytest.raises(ModeleIndisponible, match="Intégrité"):
        charger_modele(tmp_path / "modele.joblib")


def test_fichier_modele_contenu_invalide(tmp_path):
    joblib.dump({"pipeline": None}, tmp_path / "modele.joblib")
    with pytest.raises(ModeleIndisponible, match="invalide"):
        charger_modele(tmp_path / "modele.joblib")


def test_fichier_modele_variables_differentes(tmp_path):
    paquet = joblib.load(MODELE)
    paquet["variables"] = list(reversed(paquet["variables"]))
    joblib.dump(paquet, tmp_path / "modele.joblib")
    with pytest.raises(ModeleIndisponible, match="variables"):
        charger_modele(tmp_path / "modele.joblib")


def test_api_avec_modele_impossible_a_charger(client, monkeypatch, tmp_path, caplog):
    service = ServicePrediction()
    with caplog.at_level(logging.ERROR, logger="creditrisk"):
        service.charger(tmp_path / "absent.joblib")
    assert not service.disponible
    assert "Chargement du modèle impossible" in caplog.text  # tracé dans les journaux
    monkeypatch.setattr(main, "service", service)
    assert client.get("/health").status_code == 503
    reponse = client.post("/predict", json=REFERENCE)
    assert reponse.status_code == 503 and reponse.json()["erreur"] == "modele_indisponible"
    assert sans_trace_technique(reponse.text)


# ================================================================== une variable attendue est absente
def test_variable_attendue_absente_au_pretraitement(client, monkeypatch):
    def sans_duree(demande):
        return demande_vers_dataframe(demande).drop(columns="duree_mois")

    monkeypatch.setattr(prediction, "demande_vers_dataframe", sans_duree)
    verifier_500(client.post("/predict", json=REFERENCE))


# ================================================================== le prétraitement retourne une erreur
def test_pretraitement_en_erreur(client, monkeypatch, caplog):
    def en_erreur(_):
        raise PretraitementErreur("détail interne du prétraitement")

    monkeypatch.setattr(prediction, "construire_variables", en_erreur)
    with caplog.at_level(logging.ERROR, logger="creditrisk"):
        reponse = client.post("/predict", json=REFERENCE)
    verifier_500(reponse)
    assert "détail interne" not in reponse.text  # absent de la réponse…
    assert "détail interne" in caplog.text  # …mais conservé dans les journaux


# ================================================================== dimensions incorrectes
def test_dimensions_incorrectes_detectees_avant_le_modele(client, monkeypatch):
    appels = []
    pipeline = main.service.paquet["pipeline"]
    monkeypatch.setattr(prediction, "construire_variables", lambda df: construire_variables(df).iloc[:, :4])
    monkeypatch.setattr(pipeline, "predict_proba", lambda X: appels.append(X))
    verifier_500(client.post("/predict", json=REFERENCE))
    assert appels == []  # le modèle n'a pas été appelé avec des données mal dimensionnées


# ================================================================== erreur interne pendant la prédiction
def test_erreur_interne_du_modele(client, monkeypatch):
    def plante(_):
        raise RuntimeError("/home/serveur/modele.py ligne 42 : mot de passe base=secret")

    monkeypatch.setattr(main.service.paquet["pipeline"], "predict_proba", plante)
    reponse = client.post("/predict", json=REFERENCE)
    verifier_500(reponse)
    assert "secret" not in reponse.text


@pytest.mark.parametrize("valeur", [float("nan"), 1.5, -0.1])
def test_probabilite_invalide_produite_par_le_modele(client, monkeypatch, valeur):
    monkeypatch.setattr(main.service.paquet["pipeline"], "predict_proba", lambda X: np.array([[0.0, valeur]]))
    verifier_500(client.post("/predict", json=REFERENCE))


def test_service_retabli_apres_panne(client, monkeypatch):
    """Après une erreur interne, la requête suivante est traitée normalement (pas d'état corrompu)."""

    def panne(_):
        raise RuntimeError("panne")

    with monkeypatch.context() as m:
        m.setattr(prediction, "construire_variables", panne)
        assert client.post("/predict", json=REFERENCE).status_code == 500
    assert client.post("/predict", json=REFERENCE).status_code == 200
