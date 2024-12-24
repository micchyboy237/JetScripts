import json
import spacy
from spacy import displacy
from jet.logger import logger


# Explanations and real-world usage examples for the key attributes in the Span class.

def explain_span_attributes():
    """
    Explains Span attributes and provides their real-world usage.
    """
    explanations = {
        "start": "The starting token index of the span in the document.",
        "end": "The ending token index of the span in the document.",
        "start_char": "The character offset where the span starts in the text.",
        "end_char": "The character offset where the span ends in the text.",
        "label_": "The entity label as a string (e.g., PERSON, ORG).",
        "vector": "The vector representation of the span for similarity comparisons.",
        "vector_norm": "The magnitude of the vector for the span.",
        "kb_id_": "A unique identifier for linking the span to an external knowledge base.",
        "sentiment": "A sentiment score of the span based on the underlying model.",
        "similarity": "A method to compute similarity between the span and other linguistic objects.",
    }
    return explanations


def real_world_usage_examples():
    """
    Provides real-world usage examples for key Span attributes and methods.
    """
    examples = {
        "start_end_char": {
            "description": "Use start_char and end_char to highlight entities in the text.",
            "function": highlight_entities,
        },
        "label_usage": {
            "description": "Use label_ to classify named entities for further action.",
            "function": classify_entities,
        },
        "text_similarity": {
            "description": "Use similarity for clustering or matching entities.",
            "function": text_similarity,
        },
        "vector_similarity": {
            "description": "Use vector for clustering or matching entities.",
            "function": vector_norms,
        },
        "sentiment_analysis": {
            "description": "Use sentiment for context-aware actions.",
            "function": analyze_sentiment,
        },
    }
    return examples


# Example usage of attributes in Python functions.

def highlight_entities(doc):
    """
    Highlights entities in the text based on their character offsets.
    """
    highlights = {}
    for entity in doc.ents:
        highlights[entity.text] = [entity.start_char, entity.end_char]
    return highlights


def classify_entities(doc):
    """
    Classifies entities based on their labels and organizes them by type.
    """
    classified = {}
    for entity in doc.ents:
        label = entity.label_
        classified[label] = classified.get(label, []) + [entity.text]
    return classified


def text_similarity(doc):
    """
    Compares similarity between the document entities and all texts.
    """
    similarities = {
        entity.text: f"{entity.similarity(doc):.4f}" for entity in doc.ents}
    return similarities


def vector_norms(doc):
    """
    Gets vector_norm scores from each document.
    """
    norms = {entity.text: f"{entity.vector_norm:.4f}" for entity in doc.ents}
    return norms


def analyze_sentiment(doc):
    """
    Analyzes sentiment of each entity in the document.
    """
    sentiments = {entity.text: entity.sentiment for entity in doc.ents}
    return sentiments


def compare_similarity(doc, other_text):
    """
    Compares similarity between the document entities and another text.
    """
    other_doc = spacy.load("en_core_web_sm")(other_text)
    similarities = []
    for entity in doc.ents:
        for other_entity in other_doc.ents:
            similarity = entity.similarity(other_entity)
            similarities.append((entity.text, other_entity.text, similarity))
    return similarities


def main_info():
    explanations = explain_span_attributes()
    for attr, explanation in explanations.items():
        logger.log(f'"{attr}":', explanation, colors=["DEBUG", "WHITE"])


def main_features(doc):
    # Run examples
    logger.newline()
    logger.info("Real-world usage examples for Span attributes:")
    examples = real_world_usage_examples()
    for idx, (key, example) in enumerate(examples.items()):
        logger.newline()
        logger.log(f"{idx + 1}.)", f'"{key}" -', example['description'],
                   colors=["INFO", "INFO", "WHITE"])
        logger.success(json.dumps(example['function'](doc), indent=2))
        logger.log("---")


def main_compare(doc, other_text):
    """
    Compares entities in the input document with those in another text, then logs the results.
    """
    # Compare similarity between the entities of the doc and other_text
    similarities = compare_similarity(doc, other_text)

    # Log the results
    logger.newline()
    logger.info("Entity Similarity Comparison:")

    for entity1, entity2, similarity in similarities:
        logger.newline()
        logger.success(
            f"Entity 1: {entity1}",
            "|",
            f"Entity 2: {entity2}",
            "|",
            f"Similarity: {similarity:.4f}",
            colors=["INFO", "WHITE", "INFO", "WHITE", "SUCCESS"]
        )
        logger.log("---")


if __name__ == "__main__":
    # Load the spaCy model with the default NER component
    nlp = spacy.load("en_core_web_sm")

    # Add SpanMarker as an additional NER pipeline
    nlp.add_pipe("span_marker", config={
        # "model": "tomaarsen/span-marker-mbert-base-multinerd"
        "model": "tomaarsen/span-marker-roberta-large-ontonotes5"
    }, last=True)

    # Input text
    text = """Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the 
    Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her 
    death in 30 BCE."""

    # Process the text using the updated spaCy pipeline
    doc = nlp(text)

    # main_info()
    main_features(doc)

    # other_text = """Cleopatra the Great, born in 69 BCE, ruled Egypt as the last ruler of the
    # Ptolemaic Kingdom. She passed away in 30 BCE."""
    # main_compare(doc, other_text=other_text)
