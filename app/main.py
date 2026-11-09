"""API de prédiction du risque de défaut (contrat : docs/plan_de_test.md, § 2).

Lancement local : uvicorn app.main:app --reload
Le chemin du modèle peut être changé avec la variable d'environnement CREDITRISK_MODELE.
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.prediction import CHEMIN_MODELE_DEFAUT, ModeleIndisponible, ServicePrediction
from app.validation import DemandeCredit, formater_erreurs

logger = logging.getLogger("creditrisk")

app = FastAPI(
    title="CreditRisk-Test",
    version="1.1.0",
    description="Estimation du risque de défaut de remboursement d'un demandeur de crédit (prototype de test).",
)

service = ServicePrediction()
service.charger(os.environ.get("CREDITRISK_MODELE", CHEMIN_MODELE_DEFAUT))

MESSAGE_INDISPONIBLE = "Le modèle de prédiction n'est pas disponible. Réessayez plus tard."
MESSAGE_INTERNE = "Une erreur interne est survenue. La demande n'a pas été traitée."


class ReponsePrediction(BaseModel):
    probabilite_defaut: float
    niveau_risque: str
    defaut_predit: bool
    alertes: list[str]
    version_modele: str


def erreur(code: int, type_erreur: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"erreur": type_erreur, "message": message})


@app.exception_handler(RequestValidationError)
async def donnees_invalides(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 : liste {champ, message}, sans reprendre les valeurs saisies."""
    return JSONResponse(
        status_code=422,
        content={"erreur": "donnees_invalides", "details": formater_erreurs(exc.errors())},
    )


@app.get("/health")
def health():
    if not service.disponible:
        return JSONResponse(status_code=503, content={"statut": "degrade", "modele_charge": False})
    return {"statut": "ok", "modele_charge": True, "version_modele": service.paquet["metadonnees"]["version"]}


@app.get("/model-info")
def model_info():
    try:
        return service.informations()
    except ModeleIndisponible:
        return erreur(503, "modele_indisponible", MESSAGE_INDISPONIBLE)


@app.post("/predict", response_model=ReponsePrediction)
def predict(demande: DemandeCredit):
    try:
        return service.predire(demande)
    except ModeleIndisponible:
        return erreur(503, "modele_indisponible", MESSAGE_INDISPONIBLE)
    except Exception:  # prétraitement, dimensions, modèle : détail dans les journaux, jamais dans la réponse
        logger.exception("Erreur interne pendant la prédiction")
        return erreur(500, "erreur_interne", MESSAGE_INTERNE)
