import sys
import json

import spacy


def main():
    args = sys.argv[1:]  # Ignore script name

    # Read arguments
    model = sys.argv[1]
    text = sys.argv[2]
    labels = json.loads(sys.argv[3])  # Convert JSON string back to list
    style = sys.argv[4]
    chunk_size = int(sys.argv[5])  # Convert back to integer

    # Run NLP

    # Configure SpaCy with dynamic chunk size
    custom_spacy_config = {
        "gliner_model": model,
        "chunk_size": chunk_size,
        "labels": labels,
        "style": style
    }

    nlp = spacy.blank("en")
    nlp.add_pipe("gliner_spacy", config=custom_spacy_config)

    doc = nlp(text)

    entities = [{
        "text": entity.text,
        "label": entity.label_,
        "score": f"{entity._.score:.4f}"
    } for entity in doc.ents]

    print(f"result: {json.dumps(entities)}")


if __name__ == "__main__":
    main()
