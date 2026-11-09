"""Tests fonctionnels de l'API (Partie II) : contrat des endpoints /health, /predict et /model-info."""

import math

import pytest
from conftest import CHAMPS_REPONSE, REFERENCE, avec, sans_trace_technique


def champs(reponse) -> set:
    return {d["champ"] for d in reponse.json()["details"]}


def verifier_422(reponse, champ_attendu=None):
    assert reponse.status_code == 422
    corps = reponse.json()
    assert corps["erreur"] == "donnees_invalides"
    assert all({"champ", "message"} <= d.keys() for d in corps["details"])
    if champ_attendu:
        assert champ_attendu in champs(reponse)
    assert sans_trace_technique(reponse.text)


# ------------------------------------------------------------------ 1. requête valide (CT01, CT09)
def test_requete_valide(client):
    reponse = client.post("/predict", json=REFERENCE)
    assert reponse.status_code == 200
    resultat = reponse.json()
    assert set(resultat) == CHAMPS_REPONSE
    assert math.isfinite(resultat["probabilite_defaut"])
    assert 0 <= resultat["probabilite_defaut"] <= 1
    assert resultat["niveau_risque"] in {"faible", "modere", "eleve"}
    assert isinstance(resultat["defaut_predit"], bool)
    assert resultat["alertes"] == []
    assert resultat["version_modele"] == "v1"


def test_requete_valide_montants_decimaux(client):
    assert client.post("/predict", json=avec(revenu_mensuel=250000.75)).status_code == 200


# ------------------------------------------------------------------ 2. champ obligatoire absent (CT13)
@pytest.mark.parametrize("champ", list(REFERENCE))
def test_requete_sans_champ_obligatoire(client, champ):
    donnees = {k: v for k, v in REFERENCE.items() if k != champ}
    reponse = client.post("/predict", json=donnees)
    verifier_422(reponse, champ)
    assert any("manquant" in d["message"] for d in reponse.json()["details"])


def test_requete_champ_null(client):
    verifier_422(client.post("/predict", json=avec(age=None)), "age")


def test_requete_objet_vide_signale_tous_les_champs(client):
    reponse = client.post("/predict", json={})
    verifier_422(reponse)
    assert champs(reponse) == set(REFERENCE)


# ------------------------------------------------------------------ 3. âge négatif (CT02)
def test_requete_age_negatif(client):
    verifier_422(client.post("/predict", json=avec(age=-5)), "age")


@pytest.mark.parametrize("age", [17, 76])
def test_requete_age_hors_intervalle(client, age):
    verifier_422(client.post("/predict", json=avec(age=age, anciennete_activite=0)), "age")


# ------------------------------------------------------------------ 4. revenu sous forme de texte (CT04)
@pytest.mark.parametrize("revenu", ["250000", "deux cent mille"])
def test_requete_revenu_texte(client, revenu):
    reponse = client.post("/predict", json=avec(revenu_mensuel=revenu))
    verifier_422(reponse, "revenu_mensuel")
    assert revenu not in reponse.text  # la valeur saisie n'est pas renvoyée


# ------------------------------------------------------------------ 5. montant nul ou négatif (CT14)
@pytest.mark.parametrize("montant", [0, -1000])
def test_requete_montant_nul_ou_negatif(client, montant):
    verifier_422(client.post("/predict", json=avec(montant_credit=montant)), "montant_credit")


# ------------------------------------------------------------------ 6. durée non entière (CT07)
@pytest.mark.parametrize("duree", [12.5, 12.0, "12"])
def test_requete_duree_non_entiere(client, duree):
    verifier_422(client.post("/predict", json=avec(duree_mois=duree)), "duree_mois")


# ------------------------------------------------------------------ 7. champs supplémentaires (CT15)
def test_requete_champ_supplementaire(client):
    reponse = client.post("/predict", json=avec(commentaire="client fidèle"))
    verifier_422(reponse, "commentaire")
    assert "non autorisé" in reponse.json()["details"][0]["message"]


# ------------------------------------------------------------------ 8. valeurs extrêmement élevées (CT14, CT22)
@pytest.mark.parametrize(
    "modif, champ",
    [
        ({"revenu_mensuel": 10**12}, "revenu_mensuel"),
        ({"montant_credit": 10**15}, "montant_credit"),
        ({"age": 10**6}, "age"),
        ({"duree_mois": 10**9}, "duree_mois"),
    ],
)
def test_requete_valeurs_extremes_hors_limites_rejetees(client, modif, champ):
    verifier_422(client.post("/predict", json=avec(**modif)), champ)


def test_requete_valeur_elevee_mais_autorisee_signalee(client):
    reponse = client.post("/predict", json=avec(revenu_mensuel=20_000_000))
    assert reponse.status_code == 200
    assert any("revenu_mensuel" in a for a in reponse.json()["alertes"])


# ------------------------------------------------------------------ 9. modèle indisponible (CT11)
def test_predict_modele_indisponible(client, sans_modele):
    reponse = client.post("/predict", json=REFERENCE)
    assert reponse.status_code == 503
    assert reponse.json()["erreur"] == "modele_indisponible"
    assert sans_trace_technique(reponse.text)


def test_health_modele_indisponible(client, sans_modele):
    reponse = client.get("/health")
    assert reponse.status_code == 503
    assert reponse.json() == {"statut": "degrade", "modele_charge": False}


def test_model_info_modele_indisponible(client, sans_modele):
    reponse = client.get("/model-info")
    assert reponse.status_code == 503
    assert reponse.json()["erreur"] == "modele_indisponible"


def test_validation_avant_disponibilite_du_modele(client, sans_modele):
    """Une entrée invalide reste une erreur du client (422), même si le modèle est indisponible."""
    assert client.post("/predict", json=avec(age=-5)).status_code == 422


# ------------------------------------------------------------------ 10. requête mal formée (CT19)
@pytest.mark.parametrize(
    "corps, type_contenu",
    [
        ('{"age": 35,', "application/json"),
        ("", "application/json"),
        ("age=35", "application/json"),
        ("[1, 2, 3]", "application/json"),
        ("age=35&revenu_mensuel=250000", "text/plain"),
    ],
)
def test_requete_mal_formee(client, corps, type_contenu):
    reponse = client.post("/predict", content=corps, headers={"Content-Type": type_contenu})
    verifier_422(reponse)


def test_methode_non_autorisee(client):
    assert client.get("/predict").status_code == 405


# ------------------------------------------------------------------ /health et /model-info (CT25)
def test_health(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json() == {"statut": "ok", "modele_charge": True, "version_modele": "v1"}


def test_model_info(client):
    reponse = client.get("/model-info")
    assert reponse.status_code == 200
    info = reponse.json()
    attendus = {"version", "algorithme", "date_entrainement", "variables_entree", "variables_modele", "seuil_decision"}
    assert attendus <= info.keys()
    assert info["variables_entree"] == list(REFERENCE)
    assert 0 < info["seuil_decision"] < 1
    assert len(info["sha256_modele"]) == 64
