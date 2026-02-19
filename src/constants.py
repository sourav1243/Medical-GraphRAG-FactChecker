# Copyright 2025 Sourav Kumar Sharma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_EMBEDDING_DIM = 1024

DEFAULT_LLM_MODEL = "google/gemini-2.0-flash-001"

CONFIDENCE_THRESHOLDS = {
    "HIGH": 0.8,
    "MEDIUM": 0.5,
    "LOW": 0.3,
}
