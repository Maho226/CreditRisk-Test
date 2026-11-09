"""Module de prétraitement : variables attendues par le modèle, dans un ordre fixe.

La normalisation (StandardScaler) fait partie du pipeline du modèle et n'est ajustée
que sur les données d'entraînement (app/training.py).
"""

import numpy as np
import pandas as pd

VARIABLES_ENTREE = ["age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"]
VARIABLES_MODELE = ["age", "log_revenu", "log_montant", "duree_mois", "anciennete_activite", "taux_endettement"]
CIBLE = "defaut"


class PretraitementErreur(Exception):
    """Les données ne peuvent pas être transformées pour le modèle."""


def demande_vers_dataframe(demande) -> pd.DataFrame:
    """Une demande validée (DemandeCredit) → DataFrame d'une ligne, colonnes d'entrée."""
    return pd.DataFrame([demande.model_dump()], columns=VARIABLES_ENTREE)


def construire_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les variables du modèle.

    taux_endettement = (montant / durée) / revenu : mensualité hors intérêts rapportée au revenu.
    Les montants sont passés au logarithme pour réduire l'effet d'échelle.
    """
    manquantes = [c for c in VARIABLES_ENTREE if c not in df.columns]
    if manquantes:
        raise PretraitementErreur(f"Variables manquantes : {manquantes}")
    x = df[VARIABLES_ENTREE].astype("float64")
    if (x[["revenu_mensuel", "montant_credit", "duree_mois"]] <= 0).any().any():
        raise PretraitementErreur("Revenu, montant et durée doivent être strictement positifs")

    sortie = pd.DataFrame(index=df.index)
    sortie["age"] = x["age"]
    sortie["log_revenu"] = np.log(x["revenu_mensuel"])
    sortie["log_montant"] = np.log(x["montant_credit"])
    sortie["duree_mois"] = x["duree_mois"]
    sortie["anciennete_activite"] = x["anciennete_activite"]
    sortie["taux_endettement"] = (x["montant_credit"] / x["duree_mois"]) / x["revenu_mensuel"]
    if not np.isfinite(sortie.to_numpy()).all():
        raise PretraitementErreur("Valeurs non finies après transformation")
    return sortie[VARIABLES_MODELE]


def verifier_dimensions(X: pd.DataFrame, n_variables_attendu: int) -> None:
    """Vérifie que le nombre de variables correspond à celui attendu par le modèle."""
    recu = X.shape[1] if X.ndim == 2 else None
    if recu != n_variables_attendu:
        raise PretraitementErreur(f"Dimensions incompatibles : {recu} variables, {n_variables_attendu} attendues")
