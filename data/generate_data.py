"""Génère le jeu de données SYNTHÉTIQUE de demandes de crédit (reproductible).

Ces données servent à TESTER LE SYSTÈME. Elles ne permettent pas de démontrer
la performance réelle d'un modèle de crédit.

Usage : python data/generate_data.py [--n 5000] [--seed 42] [--output data/credit_synthetique.csv]
Mêmes paramètres → fichier identique au bit près (toutes les valeurs sont entières).
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

N_LIGNES = 5000
SEED = 42
OUTPUT = Path(__file__).parent / "credit_synthetique.csv"
DUREES = [3, 6, 9, 12, 18, 24, 36, 48, 60]
P_DUREES = [0.05, 0.15, 0.10, 0.25, 0.15, 0.15, 0.08, 0.04, 0.03]


def generate(n: int = N_LIGNES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 71, n)  # 18 à 70 ans
    # revenu mensuel : loi log-normale, médiane 200 000 FCFA, arrondi au millier
    revenu = np.clip(np.round(rng.lognormal(np.log(200_000), 0.6, n), -3), 30_000, 5_000_000)
    duree = rng.choice(DUREES, size=n, p=P_DUREES)
    # montant : entre 5 % et 60 % du revenu cumulé sur la durée, arrondi à 10 000 FCFA
    montant = np.clip(np.round(revenu * duree * rng.uniform(0.05, 0.60, n), -4), 50_000, 50_000_000)
    # ancienneté de l'activité (années), au plus le nombre d'années depuis 18 ans
    anciennete = np.minimum(np.floor(rng.exponential(5, n)), age - 18)

    # Règle de défaut (connue, documentée) : endettement élevé, activité récente, jeunesse, crédit long
    taux_endettement = (montant / duree) / revenu
    logit = (
        -3.2
        + 6.0 * taux_endettement
        - 0.12 * anciennete
        - 0.02 * (age - 35)
        + 0.4 * (duree > 24)
        + rng.normal(0, 0.5, n)  # part non expliquée
    )
    defaut = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "age": age.astype(int),
            "revenu_mensuel": revenu.astype(int),
            "montant_credit": montant.astype(int),
            "duree_mois": duree.astype(int),
            "anciennete_activite": anciennete.astype(int),
            "defaut": defaut,
        }
    )


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=N_LIGNES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    df = generate(args.n, args.seed)
    df.to_csv(args.output, index=False, lineterminator="\n")
    print(f"{len(df)} lignes, taux de défaut {df['defaut'].mean():.1%}, SHA-256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
