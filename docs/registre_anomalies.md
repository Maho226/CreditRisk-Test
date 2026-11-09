# Robustesse et registre des anomalies — CreditRisk-Test (Partie IV)

**Version testée :** code 1.0 (commit de la Partie III), modèle `v1` (`models/model_v1.joblib`). Les anomalies A1 à A3 ont été corrigées en version 1.1 du code (§ 3).
**Suite :** `tests/test_robustness.py`. Code 1.0 : 94 tests, 90 réussis et 4 anomalies connues marquées `xfail(strict=True)` (`reports/tests_robustesse.txt`). Code 1.1 : 99 tests, tous réussis (`reports/tests_robustesse_v1.1.txt`).

## 1. Analyse des tests de robustesse

| Catégorie | Risque visé | Comportement attendu | Résultat observé | Gravité | Correction recommandée |
|---|---|---|---|---|---|
| Valeurs manquantes (`null` pour chacun des 5 champs) | Valeur par défaut silencieuse, prédiction sur un dossier incomplet | 422 nommant le champ | Conforme (5/5) | — | — |
| Valeurs non numériques (texte, chaîne vide, `"35"`, booléen, liste, objet, sur chaque champ) | Conversion implicite (`"35"` → 35, `true` → 1) ou plantage | 422 nommant le champ | Conforme (30/30) ; le mode strict refuse aussi `"35"` et `true` | — | — |
| Valeurs non finies (`NaN`, `Infinity`, `-Infinity`, `1e400`, sur chaque champ) | Calcul du taux d'endettement impossible, probabilité `NaN` | 422 sans trace technique | Conforme (20/20) | — | — |
| Limites des intervalles (16 cas, de part et d'autre de chaque borne) | Erreur de borne (≤ au lieu de <) | Bornes incluses acceptées, valeur suivante rejetée | Conforme (16/16) | — | — |
| Entrées valides hors du domaine d'entraînement (6 cas) | Confiance excessive dans une extrapolation | 200, probabilité valide, alerte nommant chaque variable concernée | Conforme pour les variables saisies (6/6) | — | — |
| Combinaison hors domaine : mensualité = 5 × revenu | Extrapolation non signalée sur la variable la plus déterminante | 200 avec alerte | **Aucune alerte** | Majeure | **A2** |
| Schéma incorrect (objet imbriqué, liste, casse ou nom de champ différents, type composé, chaîne, nombre) | Contournement du contrat, plantage | 422 sans trace technique | Conforme (7/7) | — | — |
| Propriété : 150 demandes valides tirées au hasard (Hypothesis) | Combinaison valide produisant une erreur ou une probabilité invalide | Toujours 200, probabilité finie dans [0, 1], niveau conforme | Conforme | — | — |
| Propriété : âges hors intervalle tirés au hasard | Borne contournable | Toujours 422 | Conforme | — | — |
| Cohérence : montant croissant, revenu décroissant, ancienneté croissante | Effet contre-intuitif du modèle | Risque non décroissant / non décroissant / non croissant | Conforme | — | — |
| Interprétation de la probabilité vs décision | Information contradictoire pour l'agent | `niveau_risque = eleve` si et seulement si `defaut_predit` | **Incohérent entre 0,50 et 0,52** | Majeure | **A1** |
| Montants dérisoires (revenu ou montant de 1 FCFA) | Dossier absurde évalué | Rejet | **Accepté** (avec alerte hors domaine) | Mineure | **A3** |

## 2. Registre des anomalies

| ID | Description | Reproduction | Gravité | Statut | Correction recommandée |
|---|---|---|---|---|---|
| A1 | L'interprétation (`eleve` dès 0,50) et la décision (`defaut_predit` à partir du seuil 0,52, choisi sur la validation) ne sont pas alignées : l'agent peut lire « risque élevé » et « défaut non prédit » sur le même dossier. | Montant 1 126 000, autres valeurs de la requête de référence → probabilité 0,510, `eleve`, `defaut_predit: false` | Majeure | **Corrigée (1.1)** | Niveaux dérivés du seuil du modèle (`eleve` ⇔ probabilité ≥ seuil) |
| A2 | Les alertes ne contrôlent que les variables saisies, pas les variables dérivées. Un taux d'endettement de 5 (maximum à l'entraînement : 0,60) n'est pas signalé alors qu'il détermine la prédiction (probabilité 1,0). | Montant 15 000 000, autres valeurs de la requête de référence → aucune alerte | Majeure | **Corrigée (1.1)** | Alerte si le taux d'endettement s'écarte de plus de 4 écarts-types de la moyenne d'entraînement (au-delà de 0,97 environ) ; une règle métier sur la mensualité reste à définir avec les agents |
| A3 | Les bornes inférieures du contrat (revenu et montant > 0) acceptent des valeurs absurdes. | Revenu ou montant de 1 FCFA → 200 | Mineure | **Corrigée (1.1)** | Minimum de 10 000 FCFA pour le revenu et le montant (à confirmer avec le métier) |
| A4 | Une prédiction extrapolée est présentée comme une prédiction ordinaire : seul le champ `alertes` la distingue. | 75 ans et 60 ans d'ancienneté → probabilité 0,0006, `faible`, avec deux alertes | Mineure | Ouverte | Imposer un examen manuel lorsque `alertes` n'est pas vide (règle applicative) |
| A5 | Le modèle v1 n'atteint pas le critère de rappel sur le jeu de test : 0,670 < 0,70 (validation : 0,751). La marge de 0,05 prise sur la validation n'a pas suffi. | `python -m app.training --version v1 --evaluer-test` → `reports/evaluation_test_v1.json` | **Bloquante** | Ouverte | v2 (seuil choisi pour un rappel ≥ 0,80 sur la validation) atteint un rappel de 0,766 sur le test mais un F1 de 0,441 < 0,45 : aucune version ne satisfait tous les critères (voir `docs/rapport_non_regression.md`) |

**Gravité :** bloquante = empêche le passage en pilote ; majeure = comportement incorrect qui peut induire une mauvaise décision ; mineure = défaut contournable, sans effet direct sur la décision.

Les anomalies A1 à A3 ont d'abord été reproduites par les tests `test_A1_…`, `test_A2_…` et `test_A3_…` de `tests/test_robustness.py`, marqués `xfail(strict=True)` : leur échec attendu était visible à chaque exécution.

## 3. Corrections de la version 1.1 du code

| Anomalie | Modification | Fichier | Vérification |
|---|---|---|---|
| A1 | `interpreter(probabilite, seuil_decision)` : `eleve` commence au seuil du modèle | `app/prediction.py` | `test_A1_…` (grille de 201 probabilités, cas reproduit) ; propriété Hypothesis : `eleve` ⇔ `defaut_predit` |
| A2 | `alertes_variables_derivees` : écart du taux d'endettement mesuré avec la normalisation d'entraînement | `app/prediction.py` | `test_A2_…` : alerte pour une mensualité de 5 fois le revenu |
| A3 | Minimum de 10 000 FCFA pour le revenu et le montant | `app/validation.py` | `test_A3_…`, limites 9 999 / 10 000 |

Les marques `xfail` ont été retirées : ces tests vérifient désormais la correction. **Non-régression :** les corrections ne touchent ni le prétraitement ni le modèle ; les probabilités du lot de contrôle de la référence v1 sont inchangées (écart maximal ≤ 1e-12, tolérance 1e-6), voir `reports/non_regression_code_1.1.json`.
