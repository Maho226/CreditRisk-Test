# Données

## Jeu synthétique `credit_synthetique.csv`

**Ces données sont synthétiques. Elles servent à tester le système (validation, prétraitement, modèle, API, CI). Elles ne permettent pas de démontrer la performance réelle d'un modèle de crédit.**

| Propriété | Valeur |
|---|---|
| Générateur | `data/generate_data.py` (graine 42, 5 000 lignes) |
| Commande | `python data/generate_data.py` |
| Taux de défaut | 19,7 % |
| SHA-256 | `e880e9e2d4e2e40fcdba8d43dda57dd705e95d9c48ae4d4327bb2a47eafe8436` |

Toutes les valeurs sont entières : le fichier est identique au bit près à chaque génération (mêmes versions de Python et de NumPy, voir `requirements.txt`).

## Variables

| Variable | Description | Génération |
|---|---|---|
| `age` | Âge du demandeur (années) | uniforme entre 18 et 70 |
| `revenu_mensuel` | Revenu mensuel (FCFA) | log-normale, médiane 200 000, bornée à [30 000 ; 5 000 000], arrondie au millier |
| `montant_credit` | Montant demandé (FCFA) | 5 % à 60 % du revenu cumulé sur la durée, arrondi à 10 000, borné à [50 000 ; 50 000 000] |
| `duree_mois` | Durée du crédit (mois) | 3, 6, 9, 12, 18, 24, 36, 48 ou 60 |
| `anciennete_activite` | Ancienneté de l'activité (années) | exponentielle de moyenne 5, au plus `age − 18` |
| `defaut` | 1 = défaut, 0 = remboursement sans défaut | tirage selon la règle ci-dessous |

## Règle de défaut

Probabilité de défaut = sigmoïde de :

```
−3,2 + 6,0 × taux d'endettement − 0,12 × ancienneté − 0,02 × (âge − 35) + 0,4 × [durée > 24 mois] + bruit N(0 ; 0,5)
```

avec **taux d'endettement = (montant / durée) / revenu** (mensualité hors intérêts rapportée au revenu).

La relation entre les variables et le défaut est donc connue : on peut vérifier que le modèle apprend des effets cohérents (le risque augmente avec l'endettement, diminue avec l'ancienneté).
