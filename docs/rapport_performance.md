# Rapport de performance — CreditRisk-Test (Partie VI)

**Auteur :** Mohamed KABORE — **Version mesurée :** code 1.1, modèle `v1`
**Sources :** `reports/performance.json` (script `perf/mesure_performance.py`), `reports/locust_stats.csv` et `reports/locust.html` (Locust).

## 1. Objectif de performance et justification

L'agent de crédit saisit un dossier, puis attend le résultat. Au-delà de quelques centaines de millisecondes, l'attente devient perceptible ; une institution de microfinance traite au plus quelques centaines de dossiers par jour, saisis par quelques agents en même temps.

| Objectif (fixé dans le plan de test, § 6) | Valeur |
|---|---|
| Latence au 95e percentile (P95) de `POST /predict` | ≤ 100 ms |
| Débit | ≥ 50 requêtes/s, soit une très large marge sur le besoin |
| Erreurs sous charge progressive | aucune |

## 2. Configuration de mesure

| Élément | Valeur |
|---|---|
| Machine | Google Colab, processeur x86_64, 2 cœurs logiques, 13,6 Go de mémoire |
| Système | Linux 6.6 (glibc 2.39) |
| Logiciels | Python 3.13.18, FastAPI 0.141.1, scikit-learn 1.6.1, Uvicorn 0.54.0, Locust 2.47.0 |
| Serveur | Uvicorn, 1 processus, `127.0.0.1` (client et serveur sur la même machine) |
| Protocole | horloge `time.perf_counter` ; 50 appels d'échauffement ignorés ; 1 000 requêtes par mesure ; demandes variées tirées du lot de contrôle (200 dossiers) ; 600 requêtes par palier de charge |

## 3. Latence

| Mesure | Ce qui est chronométré | Moyenne | P50 | **P95** | P99 |
|---|---|---|---|---|---|
| Inférence du modèle | `predict_proba` sur des variables déjà préparées | 1,95 ms | 1,89 ms | **2,24 ms** | 3,98 ms |
| Pipeline sans HTTP | validation + prétraitement + modèle + interprétation | 6,25 ms | 6,08 ms | **7,47 ms** | 9,73 ms |
| API en processus | `TestClient` de FastAPI (sans réseau) | 13,18 ms | 12,69 ms | **16,39 ms** | 18,87 ms |
| API par HTTP | requêtes réelles vers Uvicorn, une à la fois | 9,68 ms | 8,78 ms | **13,23 ms** | 15,31 ms |

**Interprétation.** Le modèle ne représente qu'environ 2 ms. L'essentiel du temps vient de la validation et surtout de la construction des `DataFrame` pandas pour une seule ligne (environ 4 ms), puis de la couche HTTP et de la sérialisation JSON. Le `TestClient` est plus lent que le vrai serveur, car il simule toute la pile ASGI à chaque appel : il sert aux tests fonctionnels, pas à la mesure de performance.

## 4. Débit et charge progressive

### 4.1 Paliers de clients simultanés (`perf/mesure_performance.py`)

| Clients simultanés | Débit (req/s) | Latence moyenne | P95 | Erreurs |
|---|---|---|---|---|
| 1 | 101,2 | 9,8 ms | 13,3 ms | 0 |
| 5 | 112,3 | 44,4 ms | 68,9 ms | 0 |
| 10 | **118,9** | 83,7 ms | 120,8 ms | 0 |
| 20 | 96,5 | 206,0 ms | 334,1 ms | 0 |
| 50 | 90,8 | 531,5 ms | 1 258,1 ms | 0 |

### 4.2 Locust (agents simulés)

Montée progressive d'un utilisateur par seconde jusqu'à 20, pendant 60 s ; chaque agent attend 0,1 à 0,5 s entre deux actions.

| Requête | Nombre | Échecs | Médiane | P95 | P99 | Max | Débit |
|---|---|---|---|---|---|---|---|
| `POST /predict` | 2 772 | 0 | 19 ms | **64 ms** | 93 ms | 134 ms | 46,9 req/s |
| `GET /health` | 300 | 0 | 4 ms | 25 ms | 46 ms | 60 ms | 5,1 req/s |
| Total | 3 072 | **0** | 18 ms | 62 ms | 92 ms | 134 ms | 52,0 req/s (61,5 en fin de montée) |

Le débit Locust est limité par les pauses des agents simulés, et non par le serveur : il mesure le comportement sous une charge réaliste, alors que les paliers du § 4.1 mesurent la capacité maximale.

### 4.3 Interprétation

Le serveur **sature autour de 110 à 120 requêtes/s**. Au-delà, les requêtes attendent leur tour : la latence augmente proportionnellement au nombre de clients et le débit baisse légèrement. Le service se dégrade **sans erreur** (0 échec sur 3 000 requêtes en paliers et 3 072 avec Locust). Il y a deux causes : un seul processus Uvicorn, et un traitement limité par le verrou global de Python (GIL) sur une machine à 2 cœurs.

## 5. Respect des objectifs

| Objectif | Mesure | Respecté |
|---|---|---|
| P95 de `/predict` ≤ 100 ms (requêtes une à une) | 13,2 ms | ✅ |
| P95 ≤ 100 ms jusqu'à 5 clients simultanés | 68,9 ms | ✅ |
| P95 ≤ 100 ms avec 20 agents simulés (Locust) | 64 ms | ✅ |
| P95 ≤ 100 ms avec 10 clients simultanés sans pause | 120,8 ms | ❌ |
| Débit ≥ 50 req/s | 118,9 req/s au maximum | ✅ |
| Aucune erreur sous charge progressive | 0 erreur | ✅ |

**Conclusion.** Les objectifs sont respectés pour l'usage visé (quelques agents simultanés), avec une marge d'un facteur 7 sur la latence et d'un facteur 2 sur le débit. L'objectif de latence n'est plus tenu à partir d'une dizaine de clients envoyant des requêtes **sans aucune pause**, ce qui dépasse l'usage d'une agence. Ce point est enregistré comme anomalie mineure **A6** : utiliser plusieurs processus (`uvicorn --workers`) si le service est mutualisé entre plusieurs agences.

## 6. Validation de l'outil de mesure

La première version du script de charge créait un nouveau client HTTP pour chaque requête. Cette création coûte à elle seule plusieurs dizaines de millisecondes : le débit mesuré plafonnait à environ 20 req/s, même avec un seul client, ce qui contredisait la latence séquentielle (environ 10 ms, soit environ 100 req/s). Le script a été corrigé pour utiliser un client partagé, qui réutilise ses connexions. **Un résultat de performance incohérent doit d'abord faire suspecter l'instrument de mesure.**

## 7. Limites

- Client et serveur sur la même machine : pas de latence réseau. Une agence reliée par une connexion mobile ajouterait des dizaines, voire des centaines de millisecondes.
- Machine partagée (Colab) : les mesures varient d'une exécution à l'autre ; ce sont des ordres de grandeur.
- Charge légère et de courte durée (60 s) : ni endurance, ni montée en charge sur plusieurs processus ou machines.
- Les tests `slow` (`tests/test_performance.py`) vérifient les objectifs en processus ; ils sont exclus des Pull Requests et lancés manuellement.
