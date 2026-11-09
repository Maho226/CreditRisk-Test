# Plan de test — CreditRisk-Test

**Auteur :** Mohamed KABORE — **Système :** estimation du risque de défaut de remboursement (microfinance)
**Version du plan :** 1.0, rédigée **avant** le développement et l'évaluation : les critères ci-dessous ne sont pas ajustés aux résultats.

## 1. Objet et périmètre

Le système reçoit les informations d'un demandeur de crédit, les valide, les transforme, estime une probabilité de défaut avec un modèle d'apprentissage automatique et restitue un résultat exploitable par un agent de crédit.

| Composant | Fichier | Responsabilité |
|---|---|---|
| Validation | `app/validation.py` | Format, types, intervalles autorisés, cohérence, champs interdits |
| Prétraitement | `app/preprocessing.py` | Variables dérivées (taux d'endettement, logarithmes), ordre et dimensions attendus par le modèle |
| Modèle | `app/prediction.py` | Chargement contrôlé, probabilité de défaut, interprétation (niveau de risque), alertes |
| API | `app/main.py` | `GET /health`, `POST /predict`, `GET /model-info`, gestion des erreurs |

**Dans le périmètre :** les quatre composants, leur intégration, la robustesse, la non-régression du code et du modèle, la performance locale, l'automatisation (GitHub Actions).
**Hors périmètre :** authentification, déploiement, équité et conformité réglementaire (aucun attribut sensible ; données synthétiques), performance réelle d'un modèle de crédit.

**Données.** Jeu synthétique reproductible (`data/README.md`, 5 000 lignes, 19,7 % de défauts). Il sert à tester le système et **ne permet pas de démontrer la performance réelle d'un modèle de crédit**.

## 2. Contrat de l'API

### 2.1 Entrée de `POST /predict`

Les cinq champs sont **obligatoires** ; tout autre champ est **refusé**. Les types sont **stricts** : une chaîne `"250000"` n'est pas convertie en nombre.

| Champ | Type | Intervalle autorisé | Domaine des données (alerte au-delà, bornes exactes calculées sur le jeu d'entraînement) |
|---|---|---|---|
| `age` | entier | 18 à 75 ans | 18 à 70 |
| `revenu_mensuel` | nombre fini | > 0 et ≤ 50 000 000 FCFA | 30 000 à 1 589 000 |
| `montant_credit` | nombre fini | > 0 et ≤ 100 000 000 FCFA | 50 000 à 21 760 000 |
| `duree_mois` | entier | 1 à 120 mois | 3 à 60 |
| `anciennete_activite` | entier | 0 à (âge − 15) ans | 0 à 41 |

Règle de cohérence : l'ancienneté de l'activité ne peut pas dépasser l'âge moins 15 ans.

### 2.2 Réponses

| Situation | Code HTTP | Corps |
|---|---|---|
| Prédiction réussie | 200 | `probabilite_defaut` (0 à 1), `niveau_risque` (`faible`, `modere`, `eleve`), `defaut_predit` (booléen, selon le seuil du modèle), `alertes` (liste, vide si l'entrée est dans le domaine d'entraînement), `version_modele` |
| Entrée invalide (champ manquant, type, intervalle, cohérence, champ interdit, JSON mal formé) | 422 | `erreur: "donnees_invalides"`, `details`: liste de `{champ, message}` |
| Modèle non chargé | 503 | `erreur: "modele_indisponible"`, `message` |
| Erreur interne pendant le traitement | 500 | `erreur: "erreur_interne"`, `message` générique, **sans trace technique** |
| `GET /health`, modèle chargé / non chargé | 200 / 503 | `statut` (`ok` / `degrade`), `modele_charge` |
| `GET /model-info` | 200 / 503 | version, algorithme, date d'entraînement, variables, seuil, métriques de validation, empreinte SHA-256 du modèle et des données |

**Interprétation de la probabilité** (couche applicative) : `faible` < 0,20 ≤ `modere` < 0,50 ≤ `eleve`. Le seuil de `defaut_predit` est choisi sur le jeu de validation et enregistré avec le modèle.

## 3. Stratégie de test

| Niveau | Fichier | Contenu | Exécution |
|---|---|---|---|
| Unitaire | `tests/test_validation.py` | Règles de validation et prétraitement, sans API | chaque Pull Request |
| Fonctionnel | `tests/test_api.py` | Contrat des trois endpoints (`TestClient` de FastAPI) | chaque Pull Request |
| Intégration | `tests/test_integration.py` | Validation → prétraitement → modèle → API ; pannes simulées | chaque Pull Request |
| Robustesse | `tests/test_robustness.py` | Valeurs manquantes, non numériques, non finies, limites, hors domaine, schéma incorrect ; tests par propriétés (Hypothesis) | chaque Pull Request |
| Non-régression | `tests/test_regression.py` | Contrat, prédictions de référence, métriques vs référence figée, sauvegarde/rechargement | chaque Pull Request |
| Performance | `tests/test_performance.py`, Locust | Latence, P95, débit, charge progressive | manuel (marqueur `slow`) |

**Modèle de test versionné.** Les tests utilisent un modèle sauvegardé dans le dépôt (`models/`) avec son empreinte SHA-256 : pas de réentraînement à chaque modification. Les pannes (modèle absent, prétraitement ou prédiction en erreur, dimensions incorrectes) sont simulées par substitution de composants (`monkeypatch`).

**Méthodologie du modèle.** Séparation 60 / 20 / 20 stratifiée, graine 42 ; transformations ajustées sur l'entraînement uniquement (`Pipeline`) ; seuil choisi sur la validation ; jeu de test utilisé une seule fois pour l'évaluation finale.

## 4. Matrice de traçabilité

Priorité : **P1** = défaillance aux conséquences graves (décision de crédit erronée, système inutilisable, fuite d'information) ; **P2** = défaillance gênante mais détectable ; **P3** = confort ou qualité.

| ID | Exigence | Résultat attendu | Risque si non respectée | Priorité | Cas de test |
|---|---|---|---|---|---|
| EX01 | L'âge est compris dans l'intervalle autorisé (18 à 75) | Valeurs hors limites rejetées (422) ; 18 et 75 acceptés | Crédit accordé à un mineur ; profil absurde évalué | P1 | CT01, CT02, CT03 |
| EX02 | Le revenu est numérique, fini et positif | Texte, booléen, 0, négatif, NaN, infini → erreur contrôlée (422) | Calcul du taux d'endettement faux ou division par zéro | P1 | CT04, CT05, CT06 |
| EX03 | La durée est un entier valide (1 à 120) | 12,5, 0, négatif, texte → rejet (422) | Mensualité mal calculée ; crash du prétraitement | P1 | CT07, CT08 |
| EX04 | La prédiction retourne une probabilité valide | Nombre fini dans [0, 1] pour toute entrée valide | Décision automatique sur une valeur absurde | P1 | CT09, CT10 |
| EX05 | Le modèle est disponible | Modèle absent ou illisible → `/health` et `/predict` en 503, message clair | Service muet ou réponses inventées | P1 | CT11, CT12 |
| EX06 | Une entrée incomplète est soumise à validation | Champ manquant ou `null` → 422 indiquant le champ | Valeur par défaut silencieuse, prédiction trompeuse | P1 | CT13 |
| EX07 | Une nouvelle version ne dégrade pas les critères critiques | Comparaison à la référence figée ; régression au-delà des tolérances détectée | Mise en service d'un modèle moins bon sans le savoir | P1 | CT27, CT28, CT29 |
| EX08 | Le montant demandé est strictement positif et borné | 0, négatif, > 100 000 000 → 422 | Taux d'endettement nul ou négatif, risque sous-estimé | P1 | CT14 |
| EX09 | Les champs non prévus sont refusés | Champ supplémentaire → 422 | Erreur de saisie ignorée ; contournement du contrat | P2 | CT15 |
| EX10 | L'ancienneté est cohérente avec l'âge | Ancienneté > âge − 15 → 422 | Profil incohérent évalué comme fiable | P2 | CT16 |
| EX11 | Les erreurs n'exposent aucune trace technique | 500 / 503 / 422 sans pile d'appels, chemin de fichier ni nom de classe interne | Fuite d'information exploitable par un attaquant | P1 | CT17, CT18 |
| EX12 | Une requête mal formée est rejetée proprement | JSON invalide, corps vide, mauvais type de contenu → 422 sans plantage | Indisponibilité du service | P2 | CT19 |
| EX13 | Le prétraitement produit les variables attendues par le modèle | Variables dérivées exactes, ordre et nombre conformes ; dimensions incorrectes détectées | Prédiction sur des colonnes inversées, silencieusement fausse | P1 | CT20, CT21 |
| EX14 | Les entrées valides mais inhabituelles sont signalées | Valeur hors du domaine d'entraînement → 200 avec alerte nommant la variable | Confiance excessive dans une extrapolation | P2 | CT22 |
| EX15 | Les prédictions sont reproductibles | Même entrée → même probabilité ; écart ≤ 1e-9 après sauvegarde/rechargement | Décisions différentes pour un même dossier ; audit impossible | P1 | CT23, CT29 |
| EX16 | Les prédictions sont cohérentes avec le risque | À profil égal, la probabilité ne diminue pas quand le montant augmente ni quand le revenu diminue | Modèle aux effets contre-intuitifs, rejeté par les agents | P2 | CT24 |
| EX17 | Les métadonnées du modèle sont cohérentes | `/model-info` = version et empreinte du fichier chargé, variables du contrat | Impossible de savoir quel modèle a produit une décision | P2 | CT25 |
| EX18 | Le modèle atteint les critères minimaux sur le jeu de test | Rappel ≥ 0,70 ; F1 ≥ 0,45 ; ROC-AUC ≥ 0,70 | Défauts non détectés : pertes financières | P1 | CT26 |
| EX19 | Les objectifs de performance sont respectés | P95 de `/predict` ≤ 100 ms ; débit ≥ 50 requêtes/s ; aucune erreur sous charge progressive | Attente des agents, file de dossiers | P3 | CT30, CT31 |
| EX20 | Les tests s'exécutent automatiquement à chaque Pull Request | Ruff, Pytest et couverture ≥ 80 % ; un test défaillant fait échouer le workflow | Régression fusionnée sans contrôle | P2 | CT32 |

## 5. Cas de test

Précondition commune, sauf mention contraire : **modèle de test chargé**. Requête de référence **R** : `{"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000, "duree_mois": 12, "anciennete_activite": 4}`.

| CT | Fichier | Précondition | Données d'entrée | Résultat attendu |
|---|---|---|---|---|
| CT01 | api, validation | — | R | 200 ; corps conforme au contrat (§ 2.2) |
| CT02 | api, validation | — | R avec `age` = −5, 0, 17, 76, 150 | 422, champ `age` |
| CT03 | validation, robustesse | — | R avec `age` = 18 et 75 (limites) | Accepté |
| CT04 | api | — | R avec `revenu_mensuel` = `"250000"`, `"abc"`, `true` | 422, champ `revenu_mensuel` |
| CT05 | validation | — | R avec `revenu_mensuel` = 0, −1 | 422 |
| CT06 | robustesse | — | R avec `revenu_mensuel` = NaN, +∞ (JSON non standard) | 422, aucun plantage |
| CT07 | api | — | R avec `duree_mois` = 12,5 | 422, champ `duree_mois` |
| CT08 | validation | — | R avec `duree_mois` = 0, −12, 121, `"12"` | 422 ; 1 et 120 acceptés |
| CT09 | api | — | R | `probabilite_defaut` finie, dans [0, 1] |
| CT10 | robustesse | — | 200 entrées valides tirées au hasard dans les intervalles autorisés (Hypothesis) | Toujours 200 et probabilité valide |
| CT11 | api, intégration | Modèle non chargé | `GET /health`, `POST /predict` R | 503, `modele_charge` faux, `erreur: modele_indisponible` |
| CT12 | intégration | Fichier modèle absent, corrompu ou d'empreinte différente | Chargement | Refus du chargement, message explicite ; API en 503 |
| CT13 | api | — | R sans `age` ; R avec `age` = `null` ; `{}` | 422 nommant chaque champ manquant |
| CT14 | api | — | R avec `montant_credit` = 0, −1 000, 10¹² | 422 |
| CT15 | api | — | R + `"commentaire": "x"` | 422 |
| CT16 | validation | — | `age` = 30, `anciennete_activite` = 20 | 422 (cohérence) |
| CT17 | intégration | Prédiction forcée en erreur | R | 500, message générique, aucune trace technique |
| CT18 | api | — | Réponses 422, 500, 503 | Ni `Traceback`, ni chemin de fichier, ni nom d'exception interne |
| CT19 | api | — | Corps `{"age": 35,` ; corps vide ; texte brut | 422 |
| CT20 | intégration | — | R | Taux d'endettement = (1 000 000 / 12) / 250 000 ≈ 0,333 ; colonnes dans l'ordre attendu |
| CT21 | intégration | Prétraitement forcé en erreur ; tableau de mauvaise dimension | R | 500 contrôlé ; dimension incorrecte détectée avant le modèle |
| CT22 | robustesse | — | R avec `revenu_mensuel` = 20 000 000 ; `age` = 74 | 200 ; alerte citant la variable |
| CT23 | régression | — | R envoyée 10 fois | Probabilités identiques |
| CT24 | robustesse | — | R en faisant varier le montant (croissant) puis le revenu (décroissant) | Probabilité non décroissante |
| CT25 | api, intégration | — | `GET /model-info` | Version, variables, seuil, SHA-256 égal à celui du fichier chargé |
| CT26 | régression | Jeu de test | Évaluation finale | Critères EX18 respectés, ou rejet documenté |
| CT27 | régression | Référence figée `reports/reference_v1.json` | Prédictions sur un lot de contrôle versionné | Écart maximal de probabilité ≤ 1e-6 (même modèle, code modifié) |
| CT28 | régression | Référence figée | Métriques d'une nouvelle version sur la validation | Décision accepter / investiguer / rejeter selon les tolérances (§ 6) |
| CT29 | régression | — | Sauvegarde puis rechargement du modèle | Écart maximal ≤ 1e-9 ; empreinte vérifiée |
| CT30 | performance | API locale | 1 000 requêtes `/predict` séquentielles | Latence moyenne, P95, P99 ; P95 ≤ 100 ms |
| CT31 | performance | API lancée avec Uvicorn | Locust, charge progressive 1 → 50 utilisateurs | Débit ≥ 50 req/s, taux d'erreur nul |
| CT32 | CI | Pull Request | Workflow complet ; puis test volontairement défaillant | Succès ; puis échec bloquant la fusion |

## 6. Critères d'acceptation

**Modèle (EX18, jeu de test, seuil fixé sur la validation).** Rappel ≥ 0,70 (un défaut non détecté est une perte directe) ; F1 ≥ 0,45 (limite le refus de bons dossiers, enjeu d'inclusion financière) ; ROC-AUC ≥ 0,70. Le taux de défaut étant de 19,7 %, un modèle sans pouvoir prédictif obtient une précision de 0,20 et une AUC de 0,50.

**Non-régression (EX07), tolérances fixées avant toute comparaison :**

| Contrôle | Tolérance | Décision |
|---|---|---|
| Probabilités, même modèle après modification du code | écart maximal ≤ 1e-6 | au-delà : échec |
| Probabilités après sauvegarde/rechargement | écart maximal ≤ 1e-9 | au-delà : échec |
| Rappel (validation) | baisse ≤ 0,03 | ≤ tolérance : accepter ; ≤ 2 × tolérance : investiguer ; au-delà : rejeter |
| F1-score (validation) | baisse ≤ 0,03 | idem |
| ROC-AUC (validation) | baisse ≤ 0,02 | idem |

La référence est **figée** dans un fichier versionné : elle n'est jamais recalculée automatiquement.

**Performance (EX19).** Un agent de crédit saisit un dossier et attend la réponse : au-delà de quelques centaines de millisecondes, l'attente devient perceptible. Une agence de microfinance traite au plus quelques centaines de dossiers par jour. Objectifs, en local : **P95 de `/predict` ≤ 100 ms** et **débit ≥ 50 requêtes/s**, soit une large marge.

**Qualité.** Ruff sans erreur ; couverture du code de `app/` ≥ 80 % ; tous les tests P1 réussis.

**Distinguer invalide et inhabituel.** Une entrée **invalide** (hors des intervalles autorisés, mauvais type, incohérente) est **rejetée** (422). Une entrée **valide mais inhabituelle** (dans les intervalles autorisés mais hors du domaine d'entraînement) est **traitée** et accompagnée d'une **alerte** : la prédiction est une extrapolation à examiner.

## 7. Gravité des anomalies

| Gravité | Définition |
|---|---|
| Bloquante | Empêche le passage en pilote (exigence P1 non satisfaite, résultat faux sans signalement) |
| Majeure | Comportement incorrect contournable ou détecté |
| Mineure | Défaut de forme, de message ou de documentation |

## 8. Environnement et outils

Python 3.13, FastAPI, Pydantic, scikit-learn, Pytest, pytest-cov, Hypothesis, Ruff, Locust ; versions figées dans `requirements.txt`. Développement et exécution dans Google Colab ; intégration continue sur GitHub Actions (runner Ubuntu, CPU).
