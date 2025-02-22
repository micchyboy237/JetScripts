import sys
import json
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


def get_unique_entities(entities: List[Dict]) -> List[Dict]:
    """Ensure unique entity texts per label, keeping the highest score."""
    best_entities = {}

    for entity in entities:
        text = entity["text"]
        words = [t.lower().replace(" ", "") for t in text.split(" ") if t]
        normalized_text = " ".join(words)
        label = entity["label"]
        score = float(entity["score"])

        entity["text"] = normalized_text
        entity["score"] = score

        key = f"{label}-{str(normalized_text)}"

        if key not in best_entities or score > float(best_entities[key]["score"]):
            best_entities[key] = entity

    return list(best_entities.values())


def main():
    # Read arguments
    model = sys.argv[1]
    data = json.loads(sys.argv[2])
    labels = json.loads(sys.argv[3])
    style = sys.argv[4]

    for item in data:
        id = item['id']
        text = item['text']
        # Determine chunk size dynamically for each text
        chunk_size = determine_chunk_size(text)

        # Load SpaCy pipeline with the determined chunk size
        nlp = load_nlp_pipeline(model, labels, style, chunk_size)

        # Process the text
        doc = nlp(text)

        # Prepare the results with unique entities
        entities = get_unique_entities([{
            "text": entity.text,
            "label": entity.label_,
            "score": f"{entity._.score:.4f}"
        } for entity in doc.ents])

        # Output the result
        print(f"result: {json.dumps({
            "id": id,
            "text": text,
            "entities": entities
        })}")


if __name__ == "__main__":
    main()
