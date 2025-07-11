import os
import sys
import json
from jet.file.utils import load_file, save_file
from jet.models.model_types import LLMModelType
from jet.llm.mlx.templates.generate_labels import generate_labels
import spacy
from typing import List, Dict

# Global cache for storing the loaded pipeline
nlp_cache = None


def load_nlp_pipeline(model: str, labels: List[str], style: str, chunk_size: int):
    global nlp_cache

    if nlp_cache is None:  # Only load the model once
        custom_spacy_config = {
            "gliner_model": model,
            "chunk_size": chunk_size,
            "labels": labels,
            "style": style
        }

        nlp_cache = spacy.blank("en")
        nlp_cache.add_pipe("gliner_spacy", config=custom_spacy_config)

    return nlp_cache


def determine_chunk_size(text: str) -> int:
    """Dynamically set chunk size based on text length."""
    length = len(text)
    if length < 1000:
        return 250  # Small text, use smaller chunks
    elif length < 3000:
        return 350  # Medium text, moderate chunks
    else:
        return 500  # Large text, larger chunks


def main():
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/latest_react_web_online_jobs_philippines/docs.json"
    docs: list[dict] = load_file(docs_file)
    data = docs["documents"]

    llm_model: LLMModelType = "qwen3-1.7b-4bit"

    model = "urchade/gliner_small-v2.1"
    style = "ent"

    results = []
    for item in data:
        id = item['doc_id']
        text = f"{item["parent_header" or ""]}\n{item["header"]}\n{item["content"]}".strip()
        labels: List[str] = generate_labels(text, model_path=llm_model)
        # Determine chunk size dynamically for each text
        chunk_size = determine_chunk_size(text)

        # Load SpaCy pipeline with the determined chunk size
        nlp = load_nlp_pipeline(model, labels, style, chunk_size)

        # Process the text
        doc = nlp(text)

        # Prepare the results
        entities = [{
            "text": entity.text,
            "label": entity.label_,
            "score": f"{entity._.score:.4f}"
        } for entity in doc.ents]

        result = {
            "id": id,
            "text": text,
            "labels": labels,
            "entities": entities
        }
        # Output the result
        print(f"Result: {json.dumps(result)}")

        results.append(result)
        save_file(results, f"{output_dir}/extracted_labels.json")


if __name__ == "__main__":
    main()
