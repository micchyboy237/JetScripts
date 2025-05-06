import sys
import json
from jet.file.utils import load_file
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
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_html_myanimelist_net_Isekai/headers.json"
    data: list[dict] = load_file(data_file)
    data: list[str] = [d["text"] for d in data]
    # Read arguments
    labels = generate_labels(data, max_labels=10)
    model = "urchade/gliner_small-v2.1"
    style = "ent"

    for item in data:
        id = item['id']
        text = item['text']
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

        # Output the result
        print(f"result: {json.dumps({
            "id": id,
            "text": text,
            "entities": entities
        })}")


if __name__ == "__main__":
    main()
