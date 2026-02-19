"""
Fetch PubMedQA Medical Dataset

This module downloads and processes the PubMedQA dataset from HuggingFace,
which serves as the ground truth medical knowledge base for fact-checking.
"""

import json
import os
from datasets import load_dataset


def fetch_pubmedqa(output_path: str = "data/pubmedqa_clean.json", max_records: int = 1000):
    """
    Fetch and clean PubMedQA dataset.
    
    Args:
        output_path: Path to save cleaned JSON file
        max_records: Maximum number of records to fetch
    
    Returns:
        int: Number of records saved
    """
    print("Loading PubMedQA dataset from HuggingFace...")
    
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    
    cleaned_data = []
    for idx, item in enumerate(dataset):
        if idx >= max_records:
            break
            
        record = {
            "question": item["question"],
            "context": " ".join(item["context"]["contexts"]),
            "answer": item["final_decision"],
            "long_answer": item.get("long_answer", "")
        }
        cleaned_data.append(record)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(cleaned_data)} PubMedQA records to {output_path}")
    return len(cleaned_data)


if __name__ == "__main__":
    fetch_pubmedqa()
