"""Non-régression du logiciel et du modèle (Partie V).

Usage :
    python -m app.non_regression --creer-reference          # fige reports/reference_v1.json (une seule fois)
    python -m app.non_regression --comparer models/model_v2.joblib

La référence est figée : elle n'est jamais recalculée automatiquement (--force pour l'écraser volontairement).
Les tolérances sont fixées dans docs/plan_de_test.md (§ 6), avant toute comparaison.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.prediction import charger_modele
from app.preprocessing import CIBLE, VARIABLES_ENTREE, construire_variables
from app.training import DONNEES, metriques, separer, sha256

REFERENCE = Path("reports/reference_v1.json")
TAILLE_LOT = 200
TOLERANCE_PROBA_CODE = 1e-6  # même modèle, code modifié
TOLERANCE_PROBA_RECHARGEMENT = 1e-9  # sauvegarde puis rechargement
TOLERANCES_METRIQUES = {"rappel": 0.03, "f1": 0.03, "roc_auc": 0.02}  # baisse maximale acceptée


def lot_de_controle(chemin_donnees=DONNEES, taille: int = TAILLE_LOT) -> pd.DataFrame:
    """Demandes fixes tirées du jeu de validation (graine 42), entrées seulement."""
    _, validation, _ = separer(pd.read_csv(chemin_donnees))
    return validation[VARIABLES_ENTREE].sample(taille, random_state=42).reset_index(drop=True)


def probabilites(paquet: dict, demandes: pd.DataFrame) -> np.ndarray:
    return paquet["pipeline"].predict_proba(construire_variables(demandes))[:, 1]


def metriques_validation(paquet: dict, chemin_donnees=DONNEES) -> dict:
    _, validation, _ = separer(pd.read_csv(chemin_donnees))
    proba = probabilites(paquet, validation)
    return metriques(validation[CIBLE].to_numpy(), proba, paquet["seuil"])


def version_du_code() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnue"


def creer_reference(chemin_modele="models/model_v1.joblib", chemin_donnees=DONNEES) -> dict:
    paquet = charger_modele(chemin_modele)
    lot = lot_de_controle(chemin_donnees)
    return {
        "version_modele": paquet["metadonnees"]["version"],
        "date_creation": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_code": version_du_code(),
        "sha256_modele": paquet["sha256"],
        "sha256_donnees": sha256(chemin_donnees),
        "seuil": paquet["seuil"],
        "metriques_validation": metriques_validation(paquet, chemin_donnees),
        "tolerances": {
            "probabilite_code": TOLERANCE_PROBA_CODE,
            "probabilite_rechargement": TOLERANCE_PROBA_RECHARGEMENT,
            "baisse_metriques": TOLERANCES_METRIQUES,
        },
        "lot_controle": {
            "demandes": lot.to_dict(orient="records"),
            "probabilites": probabilites(paquet, lot).tolist(),
        },
    }


def decision_metriques(reference: dict, candidat: dict, tolerances: dict = TOLERANCES_METRIQUES) -> dict:
    """accepter si baisse ≤ tolérance ; investiguer si ≤ 2 × tolérance ; rejeter au-delà."""
    details, rang = {}, {"accepter": 0, "investiguer": 1, "rejeter": 2}
    for nom, tol in tolerances.items():
        baisse = reference[nom] - candidat[nom]
        verdict = "accepter" if baisse <= tol else "investiguer" if baisse <= 2 * tol else "rejeter"
        details[nom] = {
            "reference": reference[nom],
            "candidat": candidat[nom],
            "baisse": baisse,
            "tolerance": tol,
            "verdict": verdict,
        }
    decision = max((d["verdict"] for d in details.values()), key=rang.__getitem__)
    return {"decision": decision, "details": details}


def comparer(chemin_candidat, chemin_reference=REFERENCE, chemin_donnees=DONNEES) -> dict:
    reference = json.loads(Path(chemin_reference).read_text())
    paquet = charger_modele(chemin_candidat)
    lot = pd.DataFrame(reference["lot_controle"]["demandes"])
    proba_ref = np.array(reference["lot_controle"]["probabilites"])
    proba_cand = probabilites(paquet, lot)
    ecarts = np.abs(proba_cand - proba_ref)
    candidat = metriques_validation(paquet, chemin_donnees)
    return {
        "reference": reference["version_modele"],
        "candidat": paquet["metadonnees"]["version"],
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_code": version_du_code(),
        "ecart_probabilite_max": float(ecarts.max()),
        "ecart_probabilite_moyen": float(ecarts.mean()),
        "decisions_changees": int(np.sum((proba_cand >= paquet["seuil"]) != (proba_ref >= reference["seuil"]))),
        **decision_metriques(reference["metriques_validation"], candidat),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Référence figée et comparaison de non-régression")
    groupe = parser.add_mutually_exclusive_group(required=True)
    groupe.add_argument("--creer-reference", action="store_true")
    groupe.add_argument("--comparer", metavar="MODELE")
    parser.add_argument("--force", action="store_true", help="écraser une référence existante")
    parser.add_argument("--sortie", help="fichier du rapport de comparaison (par défaut : selon les versions)")
    args = parser.parse_args(argv)

    if args.creer_reference:
        if REFERENCE.exists() and not args.force:
            raise SystemExit(f"{REFERENCE} existe déjà : la référence n'est jamais recalculée automatiquement.")
        REFERENCE.parent.mkdir(exist_ok=True)
        REFERENCE.write_text(json.dumps(creer_reference(), indent=2, ensure_ascii=False))
        print(f"Référence figée : {REFERENCE}")
        return
    resultat = comparer(args.comparer)
    sortie = Path(args.sortie or f"reports/non_regression_{resultat['reference']}_{resultat['candidat']}.json")
    sortie.write_text(json.dumps(resultat, indent=2, ensure_ascii=False))
    print(json.dumps(resultat, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
