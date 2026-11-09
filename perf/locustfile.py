"""Test de charge léger avec Locust (Partie VI).

Serveur : uvicorn app.main:app --port 8000
Charge progressive, sans interface (1 nouvel utilisateur par seconde jusqu'à 20, pendant 60 s) :
    locust -f perf/locustfile.py --headless -u 20 -r 1 -t 60s --host http://127.0.0.1:8000 \
           --csv reports/locust --html reports/locust.html
"""

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

DEMANDES = json.loads(Path("reports/reference_v1.json").read_text())["lot_controle"]["demandes"]


class AgentDeCredit(HttpUser):
    """Un agent soumet des dossiers ; pause courte entre deux saisies pour charger le service."""

    wait_time = between(0.1, 0.5)

    @task(10)
    def evaluer_un_dossier(self):
        with self.client.post("/predict", json=random.choice(DEMANDES), catch_response=True) as reponse:
            if reponse.status_code != 200 or not 0 <= reponse.json()["probabilite_defaut"] <= 1:
                reponse.failure(f"réponse inattendue : {reponse.status_code}")

    @task(1)
    def verifier_le_service(self):
        self.client.get("/health")
