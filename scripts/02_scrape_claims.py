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
Medical Claims Scraper

This module scrapes medical claims from regional health forums and sources
in multiple languages for fact-checking validation.
"""

import json
import os
import time
from typing import Any

import requests
from requests.exceptions import ConnectionError, Timeout
from bs4 import BeautifulSoup

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

SCRAPE_TARGETS = [
    {
        "name": "Hindi Health Forum",
        "url": "https://example-hindi-health-forum.com/claims",
        "language": "hi",
        "selector": "div.post-body p",
    },
    {
        "name": "German Health Blog",
        "url": "https://example-german-health-blog.de/claims",
        "language": "de",
        "selector": "article.content p",
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MedicalResearchBot/1.0; +research)"
}


def scrape_claims(
    targets: list[dict[str, Any]] | None = None, output_path: str | None = None
) -> int:
    """
    Scrape medical claims from configured sources.

    Args:
        targets: List of target configurations
        output_path: Path to save scraped claims

    Returns:
        int: Number of claims scraped
    """
    if targets is None:
        targets = SCRAPE_TARGETS
    if output_path is None:
        output_path = str(settings.data_dir / "scraped_claims.json")

    all_claims = []

    for target in targets:
        try:
            logger.info("Scraping: %s...", target["name"])
            response = requests.get(target["url"], headers=HEADERS, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            elements = soup.select(target["selector"])

            for element in elements:
                text = element.get_text(strip=True)
                if len(text) > 30:
                    all_claims.append({
                        "claim_text": text,
                        "source_url": target["url"],
                        "language": target["language"],
                    })

            time.sleep(2)

        except Timeout:
            logger.warning("Scraping timed out for %s", target["name"])
        except ConnectionError as e:
            logger.warning("Connection failed for %s: %s", target["name"], e)
        except requests.HTTPError as e:
            logger.warning("HTTP error for %s: %s", target["name"], e)
        except Exception as e:
            logger.error(
                "Unexpected error scraping %s: %s",
                target["name"],
                e,
                exc_info=True,
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_claims, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error("Failed to write claims to %s: %s", output_path, e)
        raise

    logger.info("Saved %d claims to %s", len(all_claims), output_path)
    return len(all_claims)


def create_sample_claims(output_path: str | None = None) -> int:
    """
    Create sample claims for testing when scraping is not possible.

    Args:
        output_path: Path to save sample claims

    Returns:
        int: Number of sample claims created
    """
    if output_path is None:
        output_path = str(settings.data_dir / "scraped_claims.json")

    sample_claims = [
        {
            "claim_text": "गर्म पानी पीने से कोरोना वायरस ठीक हो जाता है।",
            "source_url": "https://example-hindi-health-forum.com/post/1",
            "language": "hi",
        },
        {
            "claim_text": "हल्दी और दूध पीने से कैंसर का इलाज होता है।",
            "source_url": "https://example-hindi-health-forum.com/post/2",
            "language": "hi",
        },
        {
            "claim_text": "नींबू का रस डायबिटीज को ठीक करता है।",
            "source_url": "https://example-hindi-health-forum.com/post/3",
            "language": "hi",
        },
        {
            "claim_text": "तुलसी के पत्ते खाने से दिल का दौरा नहीं पड़ता।",
            "source_url": "https://example-hindi-health-forum.com/post/4",
            "language": "hi",
        },
        {
            "claim_text": "Trinken von warmem Wasser heilt COVID-19.",
            "source_url": "https://example-german-health-blog.de/post/1",
            "language": "de",
        },
        {
            "claim_text": "Knoblauch kann Herzinfarkt verhindern.",
            "source_url": "https://example-german-health-blog.de/post/2",
            "language": "de",
        },
        {
            "claim_text": "Ingwer Tee behandelt Magengeschwüre.",
            "source_url": "https://example-german-health-blog.de/post/3",
            "language": "de",
        },
        {
            "claim_text": "Apfelessig senkt den Blutzucker sofort.",
            "source_url": "https://example-german-health-blog.de/post/4",
            "language": "de",
        },
        {
            "claim_text": "வெதுவெதுப்பான நீர் குடிப்பது கொரோனா வைரஸை குணப்படுத்தும்.",
            "source_url": "https://example-tamil-health.com/post/1",
            "language": "ta",
        },
        {
            "claim_text": "மஞ்சள் பால் குடிப்பது புற்றுநோயை குணப்படுத்தும்.",
            "source_url": "https://example-tamil-health.com/post/2",
            "language": "ta",
        },
        {
            "claim_text": "துளசி இலைகள் இதய நோய்களை தடுக்கின்றன.",
            "source_url": "https://example-tamil-health.com/post/3",
            "language": "ta",
        },
        {
            "claim_text": "Antibiotics are effective against viral infections like the common cold.",
            "source_url": "https://example.com/post/1",
            "language": "en",
        },
        {
            "claim_text": "Vaccination reduces the risk of severe disease and hospitalization.",
            "source_url": "https://example.com/post/2",
            "language": "en",
        },
        {
            "claim_text": "Drinking bleach cures all diseases.",
            "source_url": "https://example.com/post/3",
            "language": "en",
        },
        {
            "claim_text": "Ibuprofen treats inflammation and pain effectively.",
            "source_url": "https://example.com/post/4",
            "language": "en",
        },
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_claims, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error("Failed to write sample claims to %s: %s", output_path, e)
        raise

    logger.info("Created %d sample claims in %s", len(sample_claims), output_path)
    return len(sample_claims)


if __name__ == "__main__":
    try:
        scrape_claims()
    except Exception as e:
        logger.error("Scraping failed: %s", e)
        logger.info("Creating sample claims instead...")
        create_sample_claims()
