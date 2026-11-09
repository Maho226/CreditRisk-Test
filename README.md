# CreditRisk-Test

Stratégie de test et qualification d'un système IA d'estimation du risque de défaut de remboursement (microfinance) : validation des données, API FastAPI, tests fonctionnels, d'intégration, de robustesse, de non-régression et de performance, automatisés avec GitHub Actions.

**Auteur :** Mohamed KABORE — **Module :** Test logiciel (ESMT) — **Enseignant :** M. Moustapha DER

[![Tests CreditRisk-Test](https://github.com/Maho226/CreditRisk-Test/actions/workflows/tests.yml/badge.svg)](https://github.com/Maho226/CreditRisk-Test/actions/workflows/tests.yml)

> **Données synthétiques.** Le jeu `data/credit_synthetique.csv` est généré par un script reproductible ([`data/README.md`](data/README.md)). Il sert à tester le système et **ne permet pas de démontrer la performance réelle d'un modèle de crédit**.

## Système testé

```
POST /predict ─► validation ─► prétraitement ─► modèle ─► interprétation ─► réponse
                 (Pydantic,     (taux d'endet-    (régression  (niveau de risque,
                  types stricts) tement, log)      logistique)  alertes)
```

| Composant | Fichier | Rôle |
|---|---|---|
| Validation | `app/validation.py` | Types stricts, intervalles autorisés, cohérence âge / ancienneté, champs interdits |
| Prétraitement | `app/preprocessing.py` | Variables du modèle (taux d'endettement, logarithmes), contrôle des dimensions |
| Prédiction | `app/prediction.py` | Chargement contrôlé (SHA-256), probabilité, niveau de risque, alertes hors domaine |
| API | `app/main.py` | `GET /health`, `POST /predict`, `GET /model-info` ; erreurs 422 / 500 / 503 sans trace technique |
| Entraînement | `app/training.py` | Modèle de test versionné (`models/`), seuil choisi sur la validation |
| Non-régression | `app/non_regression.py` | Référence figée (`reports/reference_v1.json`), comparaison et décision |

Exemple :

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000, "duree_mois": 12, "anciennete_activite": 4}'
# {"probabilite_defaut": 0.4547, "niveau_risque": "modere", "defaut_predit": false, "alertes": [], "version_modele": "v1"}
```

Le contrat complet (intervalles autorisés, codes HTTP, format des erreurs) est décrit dans le [plan de test](docs/plan_de_test.md), § 2.

## Structure

```
CreditRisk-Test/
├── app/                  main.py, validation.py, preprocessing.py, prediction.py, training.py, non_regression.py
├── tests/                test_validation.py, test_api.py, test_integration.py, test_robustness.py,
│                         test_regression.py, test_performance.py (marqué « slow »), conftest.py
├── data/                 générateur et jeu synthétique reproductible (SHA-256 documenté)
├── models/               modèles de test versionnés v1 et v2 + fiches .json (empreinte, métriques)
├── perf/                 mesure_performance.py, locustfile.py
├── reports/              référence figée, évaluations, non-régression, performance, résultats des tests
├── docs/                 plan de test, registre des anomalies, rapports
├── .github/workflows/tests.yml
├── pyproject.toml        configuration Pytest et Ruff
└── requirements.txt      versions figées (Python 3.13)
```

## Exécution locale

```bash
git clone https://github.com/Maho226/CreditRisk-Test.git
cd CreditRisk-Test
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

| Action | Commande |
|---|---|
| Tests rapides (ceux de la CI) | `python -m pytest -m "not slow"` |
| Tests avec couverture | `python -m pytest -m "not slow" --cov=app --cov-fail-under=80` |
| Tests de performance | `python -m pytest -m slow` |
| Qualité du code | `ruff check . && ruff format --check .` |
| Lancer l'API | `uvicorn app.main:app --reload` puis http://127.0.0.1:8000/docs |
| Regénérer les données (même SHA-256) | `python data/generate_data.py` |
| Réentraîner un modèle | `python -m app.training --version v1` |
| Évaluation finale sur le test (une fois) | `python -m app.training --version v1 --evaluer-test` |
| Comparer un modèle à la référence | `python -m app.non_regression --comparer models/model_v2.joblib` |
| Mesures de performance | `python perf/mesure_performance.py` |
| Test de charge | `locust -f perf/locustfile.py --headless -u 20 -r 1 -t 60s --host http://127.0.0.1:8000` |

La référence de non-régression est **figée** : `python -m app.non_regression --creer-reference` refuse d'écraser `reports/reference_v1.json` (sauf `--force`).

## Intégration continue

| Job | Déclenchement | Contenu |
|---|---|---|
| `qualite` | Pull Request, push sur `main` | Ruff (règles et formatage) |
| `tests` | Pull Request, push sur `main` | Pytest hors tests lents, couverture ≥ 80 %, rapports JUnit et couverture en artefacts |
| `performance` | manuel | tests marqués `slow` |

La branche `main` est protégée : Pull Request obligatoire et réussite des deux contrôles exigée. Les tests s'appuient sur le modèle de test versionné : aucun réentraînement n'est nécessaire à chaque modification.

## Documentation

| Document | Contenu |
|---|---|
| [`docs/plan_de_test.md`](docs/plan_de_test.md) | Contrat, stratégie, matrice de traçabilité EX01–EX20, cas de test CT01–CT32, critères |
| [`docs/registre_anomalies.md`](docs/registre_anomalies.md) | Analyse de robustesse, anomalies A1–A6, corrections de la version 1.1 |
| [`docs/rapport_non_regression.md`](docs/rapport_non_regression.md) | Référence v1, code 1.1, modèle v2 |
| [`docs/rapport_performance.md`](docs/rapport_performance.md) | Latence, P95, débit, charge progressive |
