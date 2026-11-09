# Robustesse et registre des anomalies — CreditRisk-Test (Partie IV)

**Version testée :** code du commit de la Partie III, modèle `v1` (`models/model_v1.joblib`).
**Suite :** `tests/test_robustness.py`, 94 tests (90 réussis, 4 anomalies connues marquées `xfail(strict=True)`). Résultats : `reports/tests_robustesse.txt`.

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
| A1 | L'interprétation (`eleve` dès 0,50) et la décision (`defaut_predit` à partir du seuil 0,52, choisi sur la validation) ne sont pas alignées : l'agent peut lire « risque élevé » et « défaut non prédit » sur le même dossier. | Montant 1 126 000, autres valeurs de la requête de référence → probabilité 0,510, `eleve`, `defaut_predit: false` | Majeure | Ouverte | Dériver les niveaux du seuil du modèle (`eleve` ⇔ probabilité ≥ seuil) |
| A2 | Les alertes ne contrôlent que les variables saisies, pas les variables dérivées. Un taux d'endettement de 5 (maximum à l'entraînement : 0,60) n'est pas signalé alors qu'il détermine la prédiction (probabilité 1,0). | Montant 15 000 000, autres valeurs de la requête de référence → aucune alerte | Majeure | Ouverte | Contrôler aussi le domaine des variables du modèle (taux d'endettement) ; ajouter une règle métier sur la mensualité rapportée au revenu |
| A3 | Les bornes inférieures du contrat (revenu et montant > 0) acceptent des valeurs absurdes. | Revenu ou montant de 1 FCFA → 200 | Mineure | Ouverte | Fixer des minimums réalistes avec le métier (par exemple 10 000 FCFA) |
| A4 | Une prédiction extrapolée est présentée comme une prédiction ordinaire : seul le champ `alertes` la distingue. | 75 ans et 60 ans d'ancienneté → probabilité 0,0006, `faible`, avec deux alertes | Mineure | Ouverte | Imposer un examen manuel lorsque `alertes` n'est pas vide (règle applicative) |

**Gravité :** bloquante = empêche le passage en pilote ; majeure = comportement incorrect qui peut induire une mauvaise décision ; mineure = défaut contournable, sans effet direct sur la décision.

Les anomalies A1 à A3 sont reproduites automatiquement par les tests `test_A1_…`, `test_A2_…` et `test_A3_…` de `tests/test_robustness.py`, marqués `xfail(strict=True)` : leur échec attendu est visible à chaque exécution, et ils signaleront la correction en réussissant.
