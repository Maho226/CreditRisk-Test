"""Mesures de performance (Partie VI) : latence du modèle, du pipeline et de l'API, P95, débit, charge progressive.

Usage : python perf/mesure_performance.py   → reports/performance.json

Protocole : horloge time.perf_counter ; 50 appels d'échauffement ignorés ; 1 000 requêtes séquentielles par
mesure ; demandes variées tirées du lot de contrôle de la référence ; serveur Uvicorn local (1 processus).
"""

import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app, service  # noqa: E402
from app.preprocessing import construire_variables  # noqa: E402
from app.validation import valider_demande  # noqa: E402

N, ECHAUFFEMENT = 1000, 50
PALIERS = [1, 5, 10, 20, 50]  # clients simultanés (charge progressive)
REQUETES_PAR_PALIER = 600
PORT = 8765
OBJECTIFS = {"p95_api_ms": 100.0, "debit_min_req_s": 50.0}

DEMANDES = json.loads(Path("reports/reference_v1.json").read_text())["lot_controle"]["demandes"]


def stats(durees_s) -> dict:
    ms = np.asarray(durees_s) * 1000
    return {
        "n": int(len(ms)),
        "moyenne_ms": round(float(ms.mean()), 3),
        "p50_ms": round(float(np.percentile(ms, 50)), 3),
        "p95_ms": round(float(np.percentile(ms, 95)), 3),
        "p99_ms": round(float(np.percentile(ms, 99)), 3),
        "max_ms": round(float(ms.max()), 3),
    }


def chronometrer(fonction, n=N, echauffement=ECHAUFFEMENT) -> dict:
    durees = []
    for i in range(n + echauffement):
        demande = DEMANDES[i % len(DEMANDES)]
        debut = time.perf_counter()
        fonction(demande)
        if i >= echauffement:
            durees.append(time.perf_counter() - debut)
    return stats(durees)


def configuration() -> dict:
    import fastapi
    import sklearn

    return {
        "processeur": platform.processor() or platform.machine(),
        "coeurs_logiques": os.cpu_count(),
        "memoire_go": round(psutil.virtual_memory().total / 1e9, 1),
        "systeme": platform.platform(),
        "python": platform.python_version(),
        "fastapi": fastapi.__version__,
        "scikit-learn": sklearn.__version__,
        "serveur": "uvicorn, 1 processus, 127.0.0.1",
    }


def demarrer_serveur():
    serveur = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT), "--log-level", "warning"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health").status_code == 200:
                return serveur
        except httpx.TransportError:
            time.sleep(0.1)
    serveur.terminate()
    raise RuntimeError("Le serveur Uvicorn n'a pas démarré")


def palier(url: str, clients: int) -> dict:
    """`clients` clients simultanés envoient REQUETES_PAR_PALIER requêtes au total."""

    def envoyer(i):
        with httpx.Client() as c:
            debut = time.perf_counter()
            code = c.post(url, json=DEMANDES[i % len(DEMANDES)]).status_code
            return time.perf_counter() - debut, code

    debut = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        resultats = list(pool.map(envoyer, range(REQUETES_PAR_PALIER)))
    duree = time.perf_counter() - debut
    erreurs = sum(code != 200 for _, code in resultats)
    return {
        "clients": clients,
        "debit_req_s": round(REQUETES_PAR_PALIER / duree, 1),
        "erreurs": erreurs,
        **stats([d for d, _ in resultats]),
    }


def main():
    from fastapi.testclient import TestClient

    pipeline = service.paquet["pipeline"]
    X = {i: construire_variables(pd.DataFrame([d])) for i, d in enumerate(DEMANDES)}
    index = {id(d): i for i, d in enumerate(DEMANDES)}

    resultats = {"configuration": configuration(), "protocole": __doc__.strip().splitlines()[-2:]}
    resultats["inference_modele"] = chronometrer(lambda d: pipeline.predict_proba(X[index[id(d)]]))
    resultats["pipeline_sans_http"] = chronometrer(lambda d: service.predire(valider_demande(d)))
    client = TestClient(app)
    resultats["api_en_processus"] = chronometrer(lambda d: client.post("/predict", json=d))

    serveur = demarrer_serveur()
    try:
        url = f"http://127.0.0.1:{PORT}/predict"
        with httpx.Client() as http:
            resultats["api_http_sequentielle"] = chronometrer(lambda d: http.post(url, json=d))
        resultats["charge_progressive"] = [palier(url, c) for c in PALIERS]
    finally:
        serveur.terminate()

    p95 = resultats["api_http_sequentielle"]["p95_ms"]
    debit = max(p["debit_req_s"] for p in resultats["charge_progressive"])
    resultats["objectifs"] = {
        **OBJECTIFS,
        "p95_api_mesure_ms": p95,
        "debit_max_mesure_req_s": debit,
        "p95_respecte": p95 <= OBJECTIFS["p95_api_ms"],
        "debit_respecte": debit >= OBJECTIFS["debit_min_req_s"],
        "aucune_erreur_sous_charge": all(p["erreurs"] == 0 for p in resultats["charge_progressive"]),
    }
    Path("reports/performance.json").write_text(json.dumps(resultats, indent=2, ensure_ascii=False))

    print(f"{'Mesure':<28}{'moy.':>9}{'P50':>9}{'P95':>9}{'P99':>9}  (ms)")
    for cle in ("inference_modele", "pipeline_sans_http", "api_en_processus", "api_http_sequentielle"):
        s = resultats[cle]
        print(f"{cle:<28}{s['moyenne_ms']:>9.2f}{s['p50_ms']:>9.2f}{s['p95_ms']:>9.2f}{s['p99_ms']:>9.2f}")
    print(f"\n{'Clients':>8}{'req/s':>10}{'P95 ms':>10}{'erreurs':>9}")
    for p in resultats["charge_progressive"]:
        print(f"{p['clients']:>8}{p['debit_req_s']:>10.1f}{p['p95_ms']:>10.2f}{p['erreurs']:>9}")
    print("\nObjectifs :", json.dumps(resultats["objectifs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
