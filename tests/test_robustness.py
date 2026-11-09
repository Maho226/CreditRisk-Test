"""Tests de robustesse (Partie IV) : entrées limites et comportements inattendus.

Entrée INVALIDE (hors des intervalles autorisés, mauvais type, schéma incorrect) → rejetée (422).
Entrée VALIDE MAIS INHABITUELLE (hors du domaine d'entraînement) → traitée (200) avec une alerte.
Les anomalies A1 à A3, d'abord marquées `xfail(strict=True)`, sont corrigées en version 1.1 du code ;
leurs tests vérifient désormais la correction.
"""

import json
import math

import numpy as np
import pytest
from conftest import REFERENCE, avec, sans_trace_technique
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app import main
from app.prediction import interpreter

CHAMPS = list(REFERENCE)
PROFIL = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


def poster_brut(client, corps: str):
    return client.post("/predict", content=corps, headers={"Content-Type": "application/json"})


def probabilite_valide(reponse) -> bool:
    p = reponse.json()["probabilite_defaut"]
    return math.isfinite(p) and 0.0 <= p <= 1.0


# ================================================================== valeurs manquantes
@pytest.mark.parametrize("champ", CHAMPS)
def test_valeur_manquante_null_rejetee(client, champ):
    reponse = client.post("/predict", json=avec(**{champ: None}))
    assert reponse.status_code == 422
    assert champ in {d["champ"] for d in reponse.json()["details"]}


# ================================================================== valeurs non numériques
@pytest.mark.parametrize("champ", CHAMPS)
@pytest.mark.parametrize("valeur", ["abc", "", "35", True, [35], {"valeur": 35}])
def test_valeur_non_numerique_rejetee(client, champ, valeur):
    reponse = client.post("/predict", json=avec(**{champ: valeur}))
    assert reponse.status_code == 422
    assert champ in {d["champ"] for d in reponse.json()["details"]}


# ================================================================== valeurs non finies
@pytest.mark.parametrize("champ", CHAMPS)
@pytest.mark.parametrize("valeur", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_valeur_non_finie_rejetee(client, champ, valeur):
    corps = json.dumps(REFERENCE).replace(f'"{champ}": {json.dumps(REFERENCE[champ])}', f'"{champ}": {valeur}')
    reponse = poster_brut(client, corps)
    assert reponse.status_code == 422
    assert champ in {d["champ"] for d in reponse.json()["details"]}
    assert sans_trace_technique(reponse.text)


# ================================================================== valeurs aux limites des intervalles
@pytest.mark.parametrize(
    "modif, code",
    [
        ({"age": 18, "anciennete_activite": 0}, 200),
        ({"age": 17, "anciennete_activite": 0}, 422),
        ({"age": 75}, 200),
        ({"age": 76}, 422),
        ({"duree_mois": 1}, 200),
        ({"duree_mois": 0}, 422),
        ({"duree_mois": 120}, 200),
        ({"duree_mois": 121}, 422),
        ({"revenu_mensuel": 10_000}, 200),
        ({"revenu_mensuel": 9_999.99}, 422),
        ({"montant_credit": 10_000}, 200),
        ({"montant_credit": 9_999}, 422),
        ({"revenu_mensuel": 50_000_000}, 200),
        ({"revenu_mensuel": 50_000_000.01}, 422),
        ({"montant_credit": 100_000_000}, 200),
        ({"montant_credit": 100_000_001}, 422),
        ({"anciennete_activite": 0}, 200),
        ({"anciennete_activite": -1}, 422),
        ({"age": 35, "anciennete_activite": 20}, 200),  # 35 − 15 = 20 : limite autorisée
        ({"age": 35, "anciennete_activite": 21}, 422),
    ],
)
def test_valeurs_aux_limites(client, modif, code):
    reponse = client.post("/predict", json=avec(**modif))
    assert reponse.status_code == code
    if code == 200:
        assert probabilite_valide(reponse)


# ================================================================== entrées valides mais inhabituelles
@pytest.mark.parametrize(
    "modif, variables_signalees",
    [
        ({"age": 74}, {"age"}),
        ({"revenu_mensuel": 20_000_000}, {"revenu_mensuel"}),
        ({"montant_credit": 80_000_000}, {"montant_credit"}),
        ({"duree_mois": 120}, {"duree_mois"}),
        ({"duree_mois": 1}, {"duree_mois"}),
        ({"age": 75, "anciennete_activite": 60}, {"age", "anciennete_activite"}),
    ],
)
def test_entree_hors_domaine_traitee_avec_alerte(client, modif, variables_signalees):
    reponse = client.post("/predict", json=avec(**modif))
    assert reponse.status_code == 200
    assert probabilite_valide(reponse)
    alertes = " ".join(reponse.json()["alertes"])
    assert all(v in alertes for v in variables_signalees)


def test_entree_dans_le_domaine_sans_alerte(client):
    assert client.post("/predict", json=REFERENCE).json()["alertes"] == []


# ================================================================== schéma incorrect
@pytest.mark.parametrize(
    "corps",
    [
        {"demande": REFERENCE},  # objet imbriqué
        [REFERENCE],  # liste au lieu d'un objet
        {"Age": 35, **{k: v for k, v in REFERENCE.items() if k != "age"}},  # casse différente
        {"age_demandeur": 35, **{k: v for k, v in REFERENCE.items() if k != "age"}},  # nom différent
        {**REFERENCE, "revenu_mensuel": {"montant": 250000, "devise": "FCFA"}},  # type composé
        "chaine",
        35,
    ],
)
def test_schema_incorrect_rejete(client, corps):
    reponse = client.post("/predict", json=corps)
    assert reponse.status_code == 422
    assert sans_trace_technique(reponse.text)


# ================================================================== tests par propriétés (Hypothesis)
@st.composite
def demandes_valides(draw):
    age = draw(st.integers(18, 75))
    return {
        "age": age,
        "revenu_mensuel": draw(st.floats(10_000, 50_000_000, allow_nan=False, allow_infinity=False)),
        "montant_credit": draw(st.floats(10_000, 100_000_000, allow_nan=False, allow_infinity=False)),
        "duree_mois": draw(st.integers(1, 120)),
        "anciennete_activite": draw(st.integers(0, age - 15)),
    }


@PROFIL
@given(demande=demandes_valides())
def test_propriete_toute_demande_valide_donne_une_probabilite_valide(client, demande):
    reponse = client.post("/predict", json=demande)
    assert reponse.status_code == 200
    assert probabilite_valide(reponse)
    resultat = reponse.json()
    assert resultat["niveau_risque"] == interpreter(resultat["probabilite_defaut"], main.service.paquet["seuil"])
    assert (resultat["niveau_risque"] == "eleve") == resultat["defaut_predit"]


@PROFIL
@given(age=st.one_of(st.integers(max_value=17), st.integers(min_value=76)))
def test_propriete_tout_age_hors_intervalle_est_rejete(client, age):
    assert client.post("/predict", json=avec(age=age, anciennete_activite=0)).status_code == 422


# ================================================================== cohérence des prédictions (EX16)
def probabilites(client, champ, valeurs):
    return [client.post("/predict", json=avec(**{champ: v})).json()["probabilite_defaut"] for v in valeurs]


def test_risque_non_decroissant_avec_le_montant(client):
    p = probabilites(client, "montant_credit", np.linspace(50_000, 20_000_000, 40).tolist())
    assert all(b >= a - 1e-12 for a, b in zip(p, p[1:], strict=False))


def test_risque_non_decroissant_quand_le_revenu_baisse(client):
    p = probabilites(client, "revenu_mensuel", np.linspace(1_500_000, 30_000, 40).tolist())
    assert all(b >= a - 1e-12 for a, b in zip(p, p[1:], strict=False))


def test_risque_non_croissant_avec_l_anciennete(client):
    p = probabilites(client, "anciennete_activite", list(range(0, 21)))
    assert all(b <= a + 1e-12 for a, b in zip(p, p[1:], strict=False))


# ================================================================== anomalies corrigées en version 1.1 du code
def test_A1_niveau_de_risque_coherent_avec_la_decision():
    seuil = main.service.paquet["seuil"]
    for p in np.round(np.arange(0, 1.0001, 0.005), 3):
        assert (interpreter(p, seuil) == "eleve") == (p >= seuil), f"p = {p}"


def test_A1_cas_reproduit(client):
    resultat = client.post("/predict", json=avec(montant_credit=1_126_000)).json()
    assert resultat["niveau_risque"] != "eleve" and not resultat["defaut_predit"]


def test_A2_alerte_si_mensualite_superieure_au_revenu(client):
    # montant, durée et revenu sont chacun dans le domaine, mais la mensualité vaut 5 fois le revenu
    reponse = client.post("/predict", json=avec(montant_credit=15_000_000))
    assert reponse.status_code == 200
    assert any("taux_endettement" in a for a in reponse.json()["alertes"])


@pytest.mark.parametrize("modif", [{"revenu_mensuel": 1}, {"montant_credit": 1}])
def test_A3_montants_derisoires_rejetes(client, modif):
    assert client.post("/predict", json=avec(**modif)).status_code == 422
