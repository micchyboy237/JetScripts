import json
import spacy
from spacy import displacy
from jet.logger import logger
from jet.transformers import make_serializable


def main():
    # Load the spaCy model with the default NER component
    nlp = spacy.load("en_core_web_sm")

    # Add SpanMarker as an additional NER pipeline
    nlp.add_pipe("span_marker", config={
        "model": "tomaarsen/span-marker-mbert-base-multinerd"
        # "model": "tomaarsen/span-marker-roberta-large-ontonotes5"
    }, last=True)  # Ensure SpanMarker runs after spaCy's NER

    # Input text
    text = """Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the 
    Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her 
    death in 30 BCE."""

    # Process the text using the updated spaCy pipeline
    doc = nlp(text)

    logger.newline()
    logger.debug(f"Extracted Entities ({len(doc.ents)}):")
    for entity in doc.ents:
        logger.newline()
        logger.log("Text:", entity.text, colors=["WHITE", "INFO"])
        logger.log("Label:", entity.label_, colors=["WHITE", "INFO"])
        logger.log("Start:", f"{entity.start_char}",
                   colors=["WHITE", "SUCCESS"])
        logger.log("End:", f"{entity.end_char}", colors=["WHITE", "SUCCESS"])
        logger.log("Score:", f"{entity.kb_id_ if entity.kb_id_ else 'No score'}", colors=[
                   "WHITE", "SUCCESS"])
        logger.log("Sentiment:", f"{entity.sentiment}",
                   colors=["WHITE", "INFO"])
        logger.log(
            "Vector Norm:",
            f"{entity.vector_norm if entity.has_vector else 'No vector'}",
            colors=["WHITE", "INFO"]
        )
        logger.log(
            "Similarity to other entity:",
            f"{entity.similarity(doc)}",
            colors=["WHITE", "INFO"]
        )
        logger.log("---")

    # Visualize entities using displacy
    displacy.serve(doc, style="dep", port=5002)


if __name__ == "__main__":
    main()
