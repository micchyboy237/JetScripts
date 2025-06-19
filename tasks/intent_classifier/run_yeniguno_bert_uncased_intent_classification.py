import os
from typing import Dict, List, Optional, Literal, Union
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.models.model_registry.transformers.bert_model_registry import BERTModelRegistry, ONNXBERTWrapper
from transformers import PreTrainedTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

from jet.tasks.intent_classifier import classify_intents
from jet.vectors.document_types import HeaderDocument


def main(texts: List[str]):
    """Example usage of BERTModelRegistry for intent classification."""
    registry = BERTModelRegistry()
    model_id = "yeniguno/bert-uncased-intent-classification"
    features: Dict[str, Literal["cpu", "cuda", "mps", "fp16", "fp32"]] = {
        "device": "mps",
        "precision": "fp16",
    }
    try:
        model = registry.load_model(model_id, features)
        if model is None:
            logger.error(f"Failed to load model {model_id}")
            return
        tokenizer = registry.get_tokenizer(model_id)
        if tokenizer is None:
            logger.error(f"Failed to load tokenizer for {model_id}")
            return

        results = classify_intents(
            model, tokenizer, texts, batch_size=32, show_progress=True)
        for text, result in zip(texts, results):
            logger.info(
                f"\nRank {result['rank']} (Doc: {result['doc_index']}):")
            logger.log("Score:", f"{result['score']:.4f}", colors=["SUCCESS"])
            logger.log(
                "Intent:", f"{result['label']} ({result['value']})", colors=["DEBUG"])
            logger.log("Text:", f"{result['text']}", colors=["WHITE"])
        return results
    except Exception as e:
        logger.error(f"Error during intent classification: {str(e)}")
    finally:
        registry.clear()


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    query = docs["query"]
    docs = docs["documents"]
    docs = [HeaderDocument(**doc) for doc in docs]

    texts = [
        f"Header: {f"{doc["parent_header"].lstrip('#').strip()}\n" if doc["parent_header"] else ""}{doc["header"].lstrip('#').strip()}\nContent: {doc["text"]}" for doc in docs]
    results = main(texts)
    save_file({
        "counts": {label: sum(1 for r in results if r["label"] == label) for label in set(r["label"] for r in results)},
        "results": results
    }, f"{output_dir}/results.json")
