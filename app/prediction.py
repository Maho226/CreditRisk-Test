"""Chargement contrôlé du modèle, prédiction et interprétation pour l'agent de crédit."""

import hashlib
import json
import logging
import math
from pathlib import Path

import joblib

from app.preprocessing import (
    VARIABLES_ENTREE,
    VARIABLES_MODELE,
    construire_variables,
    demande_vers_dataframe,
    verifier_dimensions,
)

logger = logging.getLogger("creditrisk")

CHEMIN_MODELE_DEFAUT = Path("models/model_v1.joblib")
SEUIL_MODERE, SEUIL_ELEVE = 0.20, 0.50  # interprétation de la probabilité (plan de test, § 2.2)


class ModeleIndisponible(Exception):
    """Aucun modèle utilisable n'est chargé."""


class PredictionErreur(Exception):
    """Le modèle n'a pas produit de probabilité valide."""


def sha256(chemin) -> str:
    return hashlib.sha256(Path(chemin).read_bytes()).hexdigest()


def charger_modele(chemin=CHEMIN_MODELE_DEFAUT) -> dict:
    """Charge un modèle sauvegardé et vérifie son intégrité et sa structure.

    joblib/pickle peut exécuter du code au chargement : seuls des fichiers de confiance sont chargés,
    et l'empreinte SHA-256 enregistrée à la sauvegarde (fichier .json) doit correspondre.
    """
    chemin = Path(chemin)
    if not chemin.is_file():
        raise ModeleIndisponible(f"Fichier du modèle introuvable : {chemin.name}")
    fiche = chemin.with_suffix(".json")
    empreinte = sha256(chemin)
    if fiche.is_file() and json.loads(fiche.read_text()).get("sha256_modele") != empreinte:
        raise ModeleIndisponible("Intégrité du modèle non vérifiée (empreinte SHA-256 différente)")
    try:
        paquet = joblib.load(chemin)
    except Exception as exc:  # fichier corrompu ou d'un autre format
        raise ModeleIndisponible(f"Fichier du modèle illisible : {chemin.name}") from exc
    if not isinstance(paquet, dict) or not {"pipeline", "seuil", "variables", "metadonnees"} <= paquet.keys():
        raise ModeleIndisponible("Contenu du fichier du modèle invalide")
    if list(paquet["variables"]) != VARIABLES_MODELE:
        raise ModeleIndisponible("Le modèle attend d'autres variables que celles du prétraitement")
    paquet["sha256"] = empreinte
    return paquet


def interpreter(probabilite: float) -> str:
    if probabilite < SEUIL_MODERE:
        return "faible"
    return "modere" if probabilite < SEUIL_ELEVE else "eleve"


def alertes_hors_domaine(donnees: dict, domaine: dict) -> list[str]:
    """Variables valides mais hors du domaine observé à l'entraînement (extrapolation)."""
    alertes = []
    for variable in VARIABLES_ENTREE:
        bas, haut = domaine[variable]
        if not bas <= donnees[variable] <= haut:
            alertes.append(f"{variable} hors du domaine d'entraînement [{bas:g} ; {haut:g}]")
    return alertes


class ServicePrediction:
    """Enchaîne prétraitement, contrôle des dimensions, modèle et interprétation."""

    def __init__(self):
        self.paquet = None

    @property
    def disponible(self) -> bool:
        return self.paquet is not None

    def charger(self, chemin=CHEMIN_MODELE_DEFAUT) -> None:
        try:
            self.paquet = charger_modele(chemin)
            logger.info("Modèle %s chargé", self.paquet["metadonnees"]["version"])
        except ModeleIndisponible:
            self.paquet = None
            logger.exception("Chargement du modèle impossible")

    def predire(self, demande) -> dict:
        if not self.disponible:
            raise ModeleIndisponible("Le modèle n'est pas chargé")
        pipeline = self.paquet["pipeline"]
        X = construire_variables(demande_vers_dataframe(demande))
        verifier_dimensions(X, pipeline.n_features_in_)
        probabilite = float(pipeline.predict_proba(X)[0, 1])
        if not math.isfinite(probabilite) or not 0.0 <= probabilite <= 1.0:
            raise PredictionErreur("Probabilité invalide produite par le modèle")
        donnees = demande.model_dump()
        return {
            "probabilite_defaut": probabilite,
            "niveau_risque": interpreter(probabilite),
            "defaut_predit": probabilite >= self.paquet["seuil"],
            "alertes": alertes_hors_domaine(donnees, self.paquet["metadonnees"]["domaine_entrainement"]),
            "version_modele": self.paquet["metadonnees"]["version"],
        }

    def informations(self) -> dict:
        if not self.disponible:
            raise ModeleIndisponible("Le modèle n'est pas chargé")
        meta = self.paquet["metadonnees"]
        return {
            "version": meta["version"],
            "algorithme": meta["algorithme"],
            "date_entrainement": meta["date_entrainement"],
            "variables_entree": VARIABLES_ENTREE,
            "variables_modele": self.paquet["variables"],
            "seuil_decision": self.paquet["seuil"],
            "metriques_validation": meta["metriques_validation"],
            "sha256_modele": self.paquet["sha256"],
            "sha256_donnees": meta["donnees"]["sha256"],
        }
