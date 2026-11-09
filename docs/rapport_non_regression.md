# Rapport de non-régression — CreditRisk-Test (Partie V)

**Auteur :** Mohamed KABORE — **Référence :** version 1.0 du code, modèle `v1`

## 1. Version de référence

La référence est figée dans `reports/reference_v1.json`, créée **avant toute modification** par `python -m app.non_regression --creer-reference`. Elle n'est jamais recalculée automatiquement : un second appel est refusé, sauf avec `--force`.

| Élément enregistré | Valeur |
|---|---|
| Modèle | `v1`, SHA-256 `7f407edd…3d1b` |
| Données | `credit_synthetique.csv`, SHA-256 `e880e9e2…8436` |
| Code | commit `bab1829` |
| Seuil de décision | 0,52 |
| Métriques sur la validation (1 000 demandes, 197 défauts) | rappel 0,751 ; précision 0,394 ; F1 0,517 ; ROC-AUC 0,784 |
| Lot de contrôle | 200 demandes fixes tirées de la validation, avec leurs probabilités |

## 2. Stratégie de comparaison et tolérances

Les tolérances ont été fixées dans le plan de test (§ 6) **avant** toute comparaison.

| Contrôle | Tolérance | Test |
|---|---|---|
| Tests fonctionnels après modification du code | 100 % de réussite | suite complète (CI) |
| Sorties conformes au contrat | 200 demandes du lot | `test_sorties_conformes_au_contrat_sur_tout_le_lot` |
| Probabilités, même modèle, code modifié | écart ≤ 1e-6 | `test_probabilites_du_lot_de_controle_inchangees`, `test_probabilites_via_l_api_inchangees` |
| Sauvegarde puis rechargement | écart ≤ 1e-9 | `test_sauvegarde_puis_rechargement` |
| Baisse du rappel / du F1 / de l'AUC (validation) | 0,03 / 0,03 / 0,02 | `decision_metriques`, `test_detection_d_une_baisse_du_rappel_ou_du_f1` |

**Décision :** *accepter* si la baisse est dans la tolérance ; *investiguer* si elle est au plus du double ; *rejeter* au-delà. La capacité de détection est elle-même testée : 8 cas calculés à la main, et un modèle volontairement dégradé (seuil relevé à 0,90) qui est bien **rejeté**.

## 3. Comparaison 1 : modification du code (version 1.1)

**Modification :** correction des anomalies A1 (niveau de risque aligné sur le seuil), A2 (alerte sur le taux d'endettement) et A3 (minimum de 10 000 FCFA). Le modèle `v1` est inchangé.

| Contrôle | Résultat | Verdict |
|---|---|---|
| Suite de tests | 234 réussis (anomalies A1 à A3 désormais vérifiées comme corrigées) | conforme |
| Écart maximal de probabilité sur le lot de contrôle | 0,0 dans Colab ; 1,1 × 10⁻¹⁶ sur une autre machine | conforme (≤ 1e-6) |
| Décisions modifiées sur le lot | 0 / 200 | conforme |
| Métriques | identiques | **accepter** |

Rapport : `reports/non_regression_code_1.1.json`. Ce rapport indique le commit `bab1829`, car la comparaison a été exécutée sur la copie de travail avant le commit de la version 1.1 (`dc4de11`).

**Constat utile :** un test exigeait un écart **strictement nul**. Il réussissait dans Colab mais échouait sur une autre machine, à cause d'un écart de 10⁻¹⁶ dû aux calculs en virgule flottante. Il a été corrigé pour tolérer 10⁻¹², ce qui confirme l'intérêt de tolérances documentées plutôt que d'égalités strictes.

## 4. Comparaison 2 : nouvelle version du modèle (v2)

**Motif :** anomalie A5 : sur le jeu de test, le rappel de v1 vaut 0,670, sous le critère de 0,70.
**Modification :** même algorithme et mêmes hyperparamètres ; le seuil est choisi pour un rappel **≥ 0,80** sur la validation (au lieu de 0,75), ce qui donne un seuil de 0,43 au lieu de 0,52. Un gradient boosting sous contraintes de monotonie a aussi été essayé sur la validation : il n'apporte pas de gain (F1 de 0,46 à 0,48 pour un rappel de 0,80) et a été écarté au profit de la solution la plus simple.

| Métrique (validation) | v1 (référence) | v2 | Écart | Tolérance | Verdict |
|---|---|---|---|---|---|
| Rappel | 0,751 | 0,807 | + 0,056 | baisse ≤ 0,03 | accepter |
| F1 | 0,517 | 0,467 | **− 0,050** | baisse ≤ 0,03 | **investiguer** |
| ROC-AUC | 0,784 | 0,784 | 0 | baisse ≤ 0,02 | accepter |
| Taux de demandes signalées | 37,6 % | 48,4 % | + 10,8 points | — | — |
| Décisions modifiées sur le lot de contrôle | — | 22 / 200 | — | — | — |

**Décision de non-régression : INVESTIGUER** (`reports/non_regression_v1_v2.json`).

**Investigation.** Les probabilités sont identiques (même pipeline, écart ≤ 1e-12) : seules les décisions changent. La baisse du F1 n'est pas un défaut technique mais le **prix du compromis** : v2 détecte davantage de défauts (rappel + 0,056), mais signale près d'une demande sur deux, et la précision passe de 0,394 à 0,329. Ce compromis relève d'une **décision métier** (coût d'un défaut non détecté comparé au coût d'un bon client refusé ou d'une analyse manuelle).

**Évaluation sur le jeu de test** (`reports/evaluation_test_v2.json`) :

| Métrique (test) | v1 | v2 | Critère |
|---|---|---|---|
| Rappel | 0,670 ❌ | **0,766** ✅ | ≥ 0,70 |
| F1 | 0,470 ✅ | **0,441** ❌ | ≥ 0,45 |
| ROC-AUC | 0,741 ✅ | 0,741 ✅ | ≥ 0,70 |
| Taux signalé | 36,5 % | 48,8 % | — |

**Limite :** le jeu de test a déjà servi à évaluer v1, et c'est ce résultat qui a motivé v2. Le résultat de v2 est donc légèrement optimiste ; il faudrait un nouveau jeu jamais utilisé pour le confirmer.

## 5. Conclusion

- **Code 1.1 : accepté.** Les corrections ne modifient aucune probabilité ni aucune décision.
- **Modèle v2 : non adopté.** La non-régression a correctement signalé une baisse du F1 à investiguer, et l'évaluation finale confirme que v2 passe sous le critère F1. Aucune des deux versions ne satisfait les trois critères à la fois : avec ces variables, le modèle ne permet pas d'atteindre en même temps un rappel ≥ 0,70 et un F1 ≥ 0,45 de façon fiable.
- **Version en service :** code 1.1 avec le modèle `v1`. La non-régression reste mesurée par rapport à la référence `v1`.
- **Recommandations :** revoir les critères avec le métier, en chiffrant le coût réel des deux types d'erreur ; enrichir les données (historique de remboursement, autres engagements) ; évaluer toute nouvelle version sur des données jamais utilisées.
