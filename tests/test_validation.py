"""Tests unitaires du module de validation et du prétraitement (sans API)."""

import math

import pandas as pd
import pytest

from app.preprocessing import (
    VARIABLES_MODELE,
    PretraitementErreur,
    construire_variables,
    demande_vers_dataframe,
    verifier_dimensions,
)
from app.validation import ValidationErreur, valider_demande

REFERENCE = {"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000, "duree_mois": 12, "anciennete_activite": 4}


def avec(**modifs):
    return {**REFERENCE, **modifs}


def champs_en_erreur(donnees):
    with pytest.raises(ValidationErreur) as exc:
        valider_demande(donnees)
    return {d["champ"] for d in exc.value.details}


# ------------------------------------------------------------------ demande valide (CT01)
def test_demande_reference_valide():
    demande = valider_demande(REFERENCE)
    assert demande.age == 35 and demande.duree_mois == 12


def test_montants_decimaux_acceptes():
    assert valider_demande(avec(revenu_mensuel=250000.5)).revenu_mensuel == 250000.5


# ------------------------------------------------------------------ EX01 âge (CT02, CT03)
@pytest.mark.parametrize("age", [-5, 0, 17, 76, 150])
def test_age_hors_limites_rejete(age):
    assert "age" in champs_en_erreur(avec(age=age, anciennete_activite=0))


@pytest.mark.parametrize("age", [18, 75])
def test_age_aux_limites_accepte(age):
    assert valider_demande(avec(age=age, anciennete_activite=0)).age == age


@pytest.mark.parametrize("age", [35.0, 35.5, "35", True])
def test_age_non_entier_rejete(age):
    assert "age" in champs_en_erreur(avec(age=age))


# ------------------------------------------------------------------ EX02 revenu (CT04, CT05)
@pytest.mark.parametrize("revenu", ["250000", "abc", True, None, [250000]])
def test_revenu_non_numerique_rejete(revenu):
    assert "revenu_mensuel" in champs_en_erreur(avec(revenu_mensuel=revenu))


@pytest.mark.parametrize("revenu", [0, -1, 9_999, -250000, 50_000_001])
def test_revenu_hors_limites_rejete(revenu):
    assert "revenu_mensuel" in champs_en_erreur(avec(revenu_mensuel=revenu))


@pytest.mark.parametrize("revenu", [math.nan, math.inf, -math.inf])
def test_revenu_non_fini_rejete(revenu):
    assert "revenu_mensuel" in champs_en_erreur(avec(revenu_mensuel=revenu))


def test_revenu_maximum_accepte():
    assert valider_demande(avec(revenu_mensuel=50_000_000)).revenu_mensuel == 50_000_000


# ------------------------------------------------------------------ EX03 durée (CT07, CT08)
@pytest.mark.parametrize("duree", [12.5, 12.0, "12", 0, -12, 121])
def test_duree_invalide_rejetee(duree):
    assert "duree_mois" in champs_en_erreur(avec(duree_mois=duree))


@pytest.mark.parametrize("duree", [1, 120])
def test_duree_aux_limites_acceptee(duree):
    assert valider_demande(avec(duree_mois=duree)).duree_mois == duree


# ------------------------------------------------------------------ EX08 montant (CT14)
@pytest.mark.parametrize("montant", [0, -1000, 9_999, 100_000_001, 10**12])
def test_montant_invalide_rejete(montant):
    assert "montant_credit" in champs_en_erreur(avec(montant_credit=montant))


# ------------------------------------------------------------------ EX06, EX09 schéma (CT13, CT15)
@pytest.mark.parametrize("champ", list(REFERENCE))
def test_champ_manquant_rejete(champ):
    donnees = {k: v for k, v in REFERENCE.items() if k != champ}
    assert champ in champs_en_erreur(donnees)


def test_demande_vide_signale_tous_les_champs():
    assert champs_en_erreur({}) == set(REFERENCE)


def test_champ_supplementaire_rejete():
    assert "commentaire" in champs_en_erreur(avec(commentaire="x"))


@pytest.mark.parametrize("donnees", [None, [], "texte", 42])
def test_entree_non_objet_rejetee(donnees):
    with pytest.raises(ValidationErreur):
        valider_demande(donnees)


# ------------------------------------------------------------------ EX10 cohérence (CT16)
def test_anciennete_incoherente_avec_age_rejetee():
    with pytest.raises(ValidationErreur) as exc:
        valider_demande(avec(age=30, anciennete_activite=20))
    assert exc.value.details[0]["champ"] == "anciennete_activite"
    assert "ancienneté" in exc.value.details[0]["message"]


def test_anciennete_a_la_limite_acceptee():
    assert valider_demande(avec(age=30, anciennete_activite=15)).anciennete_activite == 15


def test_anciennete_negative_rejetee():
    assert "anciennete_activite" in champs_en_erreur(avec(anciennete_activite=-1))


# ------------------------------------------------------------------ EX11 messages sans valeurs saisies
def test_message_ne_reprend_pas_la_valeur_saisie():
    with pytest.raises(ValidationErreur) as exc:
        valider_demande(avec(revenu_mensuel="secret-123456"))
    assert "secret-123456" not in str(exc.value.details)


# ------------------------------------------------------------------ EX13 prétraitement (CT20, CT21)
def test_variables_derivees_exactes():
    X = construire_variables(demande_vers_dataframe(valider_demande(REFERENCE)))
    assert list(X.columns) == VARIABLES_MODELE
    ligne = X.iloc[0]
    assert ligne["taux_endettement"] == pytest.approx((1_000_000 / 12) / 250_000)
    assert ligne["log_revenu"] == pytest.approx(math.log(250_000))
    assert ligne["log_montant"] == pytest.approx(math.log(1_000_000))
    assert ligne["age"] == 35 and ligne["anciennete_activite"] == 4


def test_ordre_des_colonnes_d_entree_sans_effet():
    df = demande_vers_dataframe(valider_demande(REFERENCE))
    melange = df[list(reversed(df.columns))]
    pd.testing.assert_frame_equal(construire_variables(df), construire_variables(melange))


def test_pretraitement_variable_manquante():
    df = demande_vers_dataframe(valider_demande(REFERENCE)).drop(columns="duree_mois")
    with pytest.raises(PretraitementErreur, match="manquantes"):
        construire_variables(df)


def test_pretraitement_refuse_valeur_nulle():
    df = demande_vers_dataframe(valider_demande(REFERENCE))
    df.loc[0, "duree_mois"] = 0
    with pytest.raises(PretraitementErreur):
        construire_variables(df)


def test_pretraitement_refuse_valeur_non_finie():
    df = demande_vers_dataframe(valider_demande(REFERENCE)).astype(float)
    df.loc[0, "age"] = math.nan
    with pytest.raises(PretraitementErreur, match="non finies"):
        construire_variables(df)


def test_dimensions_conformes():
    X = construire_variables(demande_vers_dataframe(valider_demande(REFERENCE)))
    verifier_dimensions(X, len(VARIABLES_MODELE))


def test_dimensions_incorrectes_detectees():
    X = construire_variables(demande_vers_dataframe(valider_demande(REFERENCE)))
    with pytest.raises(PretraitementErreur, match="Dimensions"):
        verifier_dimensions(X.iloc[:, :4], len(VARIABLES_MODELE))


def test_pretraitement_sur_le_jeu_complet():
    donnees = pd.read_csv("data/credit_synthetique.csv")
    X = construire_variables(donnees)
    assert X.shape == (len(donnees), len(VARIABLES_MODELE))
    assert (X["taux_endettement"] > 0).all()
