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
Fetch PubMedQA Medical Dataset

This module downloads and processes the PubMedQA dataset from HuggingFace,
which serves as the ground truth medical knowledge base for fact-checking.
"""

import json
import os

from datasets import load_dataset

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def fetch_pubmedqa(output_path: str | None = None, max_records: int | None = None) -> int:
    """
    Fetch and clean PubMedQA dataset.

    Args:
        output_path: Path to save cleaned JSON file
        max_records: Maximum number of records to fetch

    Returns:
        int: Number of records saved
    """
    if output_path is None:
        output_path = str(settings.data_dir / "pubmedqa_clean.json")
    if max_records is None:
        max_records = settings.pubmed_sample_size

    logger.info("Loading PubMedQA dataset from HuggingFace...")

    try:
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as e:
        logger.error("Failed to load PubMedQA dataset: %s", e, exc_info=True)
        raise

    cleaned_data = []
    for idx, item in enumerate(dataset):
        if idx >= max_records:
            break

        record = {
            "question": item["question"],
            "context": " ".join(item["context"]["contexts"]),
            "answer": item["final_decision"],
            "long_answer": item.get("long_answer", ""),
            "label": item.get("label", "unknown"),
        }
        cleaned_data.append(record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error("Failed to write PubMedQA data to %s: %s", output_path, e)
        raise

    logger.info("Saved %d PubMedQA records to %s", len(cleaned_data), output_path)
    return len(cleaned_data)


if __name__ == "__main__":
    fetch_pubmedqa()
