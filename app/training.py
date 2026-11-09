"""Entraînement, sauvegarde et évaluation du modèle de test versionné.

Usage :
    python -m app.training --version v1              # entraîne et sauvegarde models/model_v1.joblib
    python -m app.training --version v1 --evaluer-test   # évaluation finale (jeu de test, une seule fois)

Séparation 60 / 20 / 20 stratifiée (graine 42) ; normalisation ajustée sur l'entraînement uniquement ;
hyperparamètre et seuil choisis sur la validation ; jeu de test réservé à l'évaluation finale.
"""

import argparse
import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.preprocessing import CIBLE, VARIABLES_ENTREE, VARIABLES_MODELE, construire_variables

SEED = 42
DONNEES = Path("data/credit_synthetique.csv")
DOSSIER_MODELES = Path("models")
warnings.filterwarnings("ignore", message="Unknown solver options")  # avertissement SciPy sans effet
RAPPEL_MIN_VALIDATION = 0.75  # marge de 0,05 au-dessus du critère d'acceptation (0,70)

VERSIONS = {
    "v1": {
        "algorithme": "Régression logistique (class_weight='balanced')",
        "grille_C": [0.01, 0.1, 1.0, 10.0],
        "rappel_min_validation": RAPPEL_MIN_VALIDATION,
    },
    # v2 : même algorithme, seuil choisi avec une marge plus large sur le rappel (anomalie A5)
    "v2": {
        "algorithme": "Régression logistique (class_weight='balanced')",
        "grille_C": [0.01, 0.1, 1.0, 10.0],
        "rappel_min_validation": 0.80,
    },
}


def sha256(chemin) -> str:
    return hashlib.sha256(Path(chemin).read_bytes()).hexdigest()


def separer(donnees: pd.DataFrame, seed: int = SEED):
    """Entraînement / validation / test (60 / 20 / 20), stratifié sur le défaut."""
    reste, test = train_test_split(donnees, test_size=0.2, stratify=donnees[CIBLE], random_state=seed)
    entrainement, validation = train_test_split(reste, test_size=0.25, stratify=reste[CIBLE], random_state=seed)
    return entrainement, validation, test


def construire_pipeline(C: float, seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("normalisation", StandardScaler()),
            ("modele", LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)),
        ]
    )


def metriques(y, proba, seuil: float) -> dict:
    pred = (proba >= seuil).astype(int)
    return {
        "rappel": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "taux_signalement": float(pred.mean()),
        "n": int(len(y)),
        "n_defauts": int(np.sum(y)),
    }


def choisir_seuil(y, proba, rappel_min: float = RAPPEL_MIN_VALIDATION) -> float:
    """Seuil maximisant le F1 parmi ceux qui garantissent rappel >= rappel_min (validation)."""
    meilleur, meilleur_f1 = 0.5, -1.0
    for seuil in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (proba >= seuil).astype(int)
        if recall_score(y, pred) >= rappel_min:
            f1 = f1_score(y, pred)
            if f1 > meilleur_f1:
                meilleur, meilleur_f1 = float(seuil), f1
    return meilleur


def environnement() -> dict:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__}


def entrainer(version: str = "v1", chemin_donnees=DONNEES, seed: int = SEED) -> dict:
    """Entraîne une version ; renvoie le paquet à sauvegarder (pipeline, seuil, métadonnées)."""
    config = VERSIONS[version]
    donnees = pd.read_csv(chemin_donnees)
    entrainement, validation, _ = separer(donnees, seed)
    X_tr, y_tr = construire_variables(entrainement), entrainement[CIBLE].to_numpy()
    X_va, y_va = construire_variables(validation), validation[CIBLE].to_numpy()

    resultats = []
    for C in config["grille_C"]:
        pipeline = construire_pipeline(C, seed).fit(X_tr, y_tr)
        resultats.append((roc_auc_score(y_va, pipeline.predict_proba(X_va)[:, 1]), C, pipeline))
    _, C, pipeline = max(resultats, key=lambda r: r[0])
    proba_va = pipeline.predict_proba(X_va)[:, 1]
    seuil = choisir_seuil(y_va, proba_va, config["rappel_min_validation"])

    return {
        "pipeline": pipeline,
        "seuil": seuil,
        "variables": VARIABLES_MODELE,
        "metadonnees": {
            "version": version,
            "algorithme": config["algorithme"],
            "hyperparametres": {"C": C},
            "rappel_min_validation": config["rappel_min_validation"],
            "date_entrainement": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "graine": seed,
            "donnees": {"fichier": Path(chemin_donnees).name, "sha256": sha256(chemin_donnees), "lignes": len(donnees)},
            "domaine_entrainement": {
                v: [float(entrainement[v].min()), float(entrainement[v].max())] for v in VARIABLES_ENTREE
            },
            "metriques_validation": metriques(y_va, proba_va, seuil),
            "environnement": environnement(),
        },
    }


def sauvegarder(paquet: dict, chemin) -> Path:
    """Sauvegarde le modèle et un fichier .json lisible contenant son empreinte SHA-256."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquet, chemin)
    fiche = {**paquet["metadonnees"], "seuil": paquet["seuil"], "variables": paquet["variables"]}
    fiche["sha256_modele"] = sha256(chemin)
    chemin.with_suffix(".json").write_text(json.dumps(fiche, indent=2, ensure_ascii=False))
    return chemin


def evaluer_test(chemin_modele, chemin_donnees=DONNEES, seed: int = SEED) -> dict:
    """Évaluation finale sur le jeu de test (à n'exécuter qu'une fois par version)."""
    paquet = joblib.load(chemin_modele)
    _, _, test = separer(pd.read_csv(chemin_donnees), seed)
    proba = paquet["pipeline"].predict_proba(construire_variables(test))[:, 1]
    return metriques(test[CIBLE].to_numpy(), proba, paquet["seuil"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Entraînement du modèle de test versionné")
    parser.add_argument("--version", default="v1", choices=sorted(VERSIONS))
    parser.add_argument("--evaluer-test", action="store_true", help="évaluation finale sur le jeu de test")
    args = parser.parse_args(argv)
    chemin = DOSSIER_MODELES / f"model_{args.version}.joblib"
    if args.evaluer_test:
        resultat = evaluer_test(chemin)
        sortie = Path("reports") / f"evaluation_test_{args.version}.json"
        sortie.parent.mkdir(exist_ok=True)
        sortie.write_text(json.dumps(resultat, indent=2, ensure_ascii=False))
        print(json.dumps(resultat, indent=2, ensure_ascii=False))
        return
    paquet = entrainer(args.version)
    sauvegarder(paquet, chemin)
    print(
        f"Modèle sauvegardé : {chemin} (seuil {paquet['seuil']}, C = {paquet['metadonnees']['hyperparametres']['C']})"
    )
    print(json.dumps(paquet["metadonnees"]["metriques_validation"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
