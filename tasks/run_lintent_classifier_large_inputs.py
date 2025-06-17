import os
from typing import Dict, List, Optional, Literal, Union
from jet.file.utils import load_file
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
    model_id = "Falconsai/intent_classification"
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
                f"Text: {text}, Intent: {result['label']}, Value: {result['value']}, Score: {result['score']:.4f}")
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

    headers = [doc["header"].lstrip('#').strip() for doc in docs]
    main(headers)
