"""Module de validation : format et cohérence des données d'un demandeur (docs/plan_de_test.md, § 2.1).

Types stricts (une chaîne "250000" n'est pas convertie), champs supplémentaires interdits,
intervalles autorisés et règle de cohérence âge / ancienneté. Les messages d'erreur ne
reprennent jamais les valeurs saisies.
"""

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

AGE_MIN, AGE_MAX = 18, 75
REVENU_MAX = 50_000_000
MONTANT_MAX = 100_000_000
DUREE_MIN, DUREE_MAX = 1, 120
AGE_DEBUT_ACTIVITE = 15  # ancienneté maximale = âge − 15


class DemandeCredit(BaseModel):
    """Données d'un demandeur de crédit."""

    model_config = ConfigDict(extra="forbid", strict=True)

    age: int = Field(ge=AGE_MIN, le=AGE_MAX, description="Âge du demandeur (années)")
    revenu_mensuel: float = Field(gt=0, le=REVENU_MAX, allow_inf_nan=False, description="Revenu mensuel (FCFA)")
    montant_credit: float = Field(gt=0, le=MONTANT_MAX, allow_inf_nan=False, description="Montant demandé (FCFA)")
    duree_mois: int = Field(ge=DUREE_MIN, le=DUREE_MAX, description="Durée du crédit (mois)")
    anciennete_activite: int = Field(ge=0, description="Ancienneté de l'activité (années)")

    @model_validator(mode="after")
    def anciennete_coherente(self):
        if self.anciennete_activite > self.age - AGE_DEBUT_ACTIVITE:
            raise ValueError(f"l'ancienneté de l'activité ne peut pas dépasser l'âge moins {AGE_DEBUT_ACTIVITE} ans")
        return self


class ValidationErreur(Exception):
    """Entrée invalide ; `details` est une liste de {champ, message}."""

    def __init__(self, details: list[dict]):
        super().__init__("Données invalides")
        self.details = details


def _message(err: dict) -> str:
    ctx = err.get("ctx", {})
    messages = {
        "missing": "champ obligatoire manquant",
        "extra_forbidden": "champ non autorisé",
        "int_type": "doit être un nombre entier",
        "float_type": "doit être un nombre",
        "finite_number": "doit être un nombre fini",
        "greater_than": f"doit être strictement supérieur à {ctx.get('gt')}",
        "greater_than_equal": f"doit être supérieur ou égal à {ctx.get('ge')}",
        "less_than_equal": f"doit être inférieur ou égal à {ctx.get('le')}",
        "json_invalid": "corps de requête JSON mal formé",
        "model_type": "un objet JSON est attendu",
        "model_attributes_type": "un objet JSON est attendu",
    }
    if err["type"] == "value_error":
        return str(ctx.get("error", "valeur invalide"))
    return messages.get(err["type"], "valeur invalide")


def formater_erreurs(erreurs: list[dict]) -> list[dict]:
    """Convertit les erreurs Pydantic/FastAPI en {champ, message}, sans les valeurs saisies."""
    details = []
    for err in erreurs:
        loc = [str(p) for p in err.get("loc", ()) if p not in ("body",)]
        details.append({"champ": ".".join(loc) or "requete", "message": _message(err)})
    return details


def valider_demande(donnees) -> DemandeCredit:
    """Valide un dictionnaire de données ; lève ValidationErreur si une règle n'est pas respectée."""
    try:
        return DemandeCredit.model_validate(donnees)
    except ValidationError as exc:
        raise ValidationErreur(formater_erreurs(exc.errors())) from None
