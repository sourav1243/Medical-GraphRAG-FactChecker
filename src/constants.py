"""
Constants Module

Defines constants, enumerations, and default values used across the application.
"""

from enum import Enum


class Verdict(str, Enum):
    """Fact-check verdict types."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    ERROR = "ERROR"


class EntityType(str, Enum):
    """Medical entity types."""
    DISEASE = "Disease"
    DRUG = "Drug"
    SYMPTOM = "Symptom"
    TREATMENT = "Treatment"
    GENE = "Gene"
    PROTEIN = "Protein"
    ANATOMICAL = "Anatomical Structure"


class RelationType(str, Enum):
    """Medical relationship types."""
    TREATS = "TREATS"
    CAUSES = "CAUSES"
    PREVENTS = "PREVENTS"
    DIAGNOSES = "DIAGNOSES"
    INTERACTS_WITH = "INTERACTS_WITH"
    LOCATED_IN = "LOCATED_IN"
    INFECTS = "INFECTS"


SUPPORTED_LANGUAGES = ["en", "hi", "de", "ta", "te", "es", "fr"]

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_LLM_MODEL = "google/gemini-2.0-flash-001"

CONFIDENCE_THRESHOLDS = {
    "HIGH": 0.8,
    "MEDIUM": 0.5,
    "LOW": 0.3
}
