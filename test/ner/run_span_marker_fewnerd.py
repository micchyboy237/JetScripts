from dataclasses import dataclass
from spacy.tokens import Doc
import json
from jet.logger import logger
import spacy
from spacy import displacy
from spacy.tokens.doc import Doc
from span_marker import SpanMarkerModel
from typing import Dict, List, Optional


def process_text(text: str, nlp: spacy.language.Language) -> Doc:
    """Process text with spaCy pipeline."""
    return nlp(text)


def log_entities(doc: Doc) -> None:
    """Log named entities with relevant details."""
    logger.newline()
    logger.debug(f"Extracted Entities ({len(doc.ents)}):")
    for entity in doc.ents:
        logger.newline()
        logger.log("Text:", entity.text, colors=["WHITE", "INFO"])
        logger.log("Label:", entity.label_, colors=["WHITE", "INFO"])
        logger.log("Start:", f"{entity.start_char}",
                   colors=["WHITE", "SUCCESS"])
        logger.log("End:", f"{entity.end_char}", colors=["WHITE", "SUCCESS"])
        score = getattr(entity, "score", "No score")
        logger.log("Score:", f"{score}", colors=["WHITE", "SUCCESS"])
        logger.log("Vector Norm:", f"{entity.vector_norm if entity.has_vector else 'No vector'}",
                   colors=["WHITE", "INFO"])
        logger.log("---")


def log_noun_chunks(doc: Doc) -> None:
    """Log noun chunks with relevant details."""
    logger.newline()
    logger.debug(f"Extracted Noun Chunks ({len(list(doc.noun_chunks))}):")
    for chunk in doc.noun_chunks:
        logger.newline()
        logger.log("Text:", chunk.text, colors=["WHITE", "INFO"])
        logger.log("Root Text:", chunk.root.text, colors=["WHITE", "INFO"])
        logger.log("Root Dependency:", chunk.root.dep_,
                   colors=["WHITE", "SUCCESS"])
        logger.log("Root Head Text:", chunk.root.head.text,
                   colors=["WHITE", "SUCCESS"])
        logger.log("---")


def log_sentences(doc: Doc) -> None:
    """Log sentences with relevant details."""
    logger.newline()
    logger.debug(f"Extracted Sentences ({len(list(doc.sents))}):")
    for i, sent in enumerate(doc.sents, 1):
        logger.newline()
        logger.log(f"Sentence {i}:", sent.text, colors=["WHITE", "INFO"])
        logger.log("Start Char:", f"{sent.start_char}", colors=[
                   "WHITE", "SUCCESS"])
        logger.log("End Char:", f"{sent.end_char}",
                   colors=["WHITE", "SUCCESS"])
        logger.log("Token Count:", f"{len(sent)}", colors=["WHITE", "SUCCESS"])
        logger.log("---")


@dataclass
class DocSentence:
    text: str
    start_char: int
    end_char: int
    token_count: int


@dataclass
class DocEntity:
    text: str
    label: str
    start_char: int
    end_char: int
    score: str
    vector_norm: str


@dataclass
class DocNounChunk:
    text: str
    root_text: str
    root_dep: str
    root_head_text: str


@dataclass
class DocSettings:
    lang: str
    direction: str


def parse_entities(doc: Doc) -> List[DocEntity]:
    """Parse a spaCy Doc into a list of DocEntity objects containing entity details.

    Args:
        doc (Doc): The spaCy document to parse.

    Returns:
        List[DocEntity]: A list of DocEntity objects with entity details.
    """
    return [
        DocEntity(
            text=entity.text,
            label=entity.label_,
            start_char=entity.start_char,
            end_char=entity.end_char,
            score=str(getattr(entity, "score", "No score")),
            vector_norm=str(
                entity.vector_norm if entity.has_vector else "No vector")
        )
        for entity in doc.ents
    ]


def parse_dependencies(doc: Doc) -> List[DocNounChunk]:
    """Parse a spaCy Doc into a list of DocNounChunk objects containing noun chunk details.

    Args:
        doc (Doc): The spaCy document to parse.

    Returns:
        List[DocNounChunk]: A list of DocNounChunk objects with noun chunk details.
    """
    return [
        DocNounChunk(
            text=chunk.text,
            root_text=chunk.root.text,
            root_dep=chunk.root.dep_,
            root_head_text=chunk.root.head.text
        )
        for chunk in doc.noun_chunks
    ]


def parse_sentences(doc: Doc) -> List[DocSentence]:
    """Parse a spaCy Doc into a list of DocSentence objects containing sentence details.

    Args:
        doc (Doc): The spaCy document to parse.

    Returns:
        List[DocSentence]: A list of DocSentence objects with sentence details.
    """
    return [
        DocSentence(
            text=sent.text,
            start_char=sent.start_char,
            end_char=sent.end_char,
            token_count=len(sent)
        )
        for sent in doc.sents
    ]


def parse_settings(doc: Doc) -> DocSettings:
    """Parse a spaCy Doc's settings into a DocSettings object.

    Args:
        doc (Doc): The spaCy document with a settings attribute.

    Returns:
        Optional[DocSettings]: A DocSettings object with language and direction, or None if settings are not available.
    """
    return DocSettings(
        lang=doc.lang_,
        direction=doc.vocab.writing_system.get("direction", "ltr"),
    )


def main():
    from jet.file.utils import save_file
    import os

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Add SpanMarker as an additional NER pipeline
    nlp.add_pipe("span_marker", config={
        "model": "tomaarsen/span-marker-bert-base-fewnerd-fine-super"
    }, last=True)

    # Input text
    text = """Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the 
    Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her 
    death in 30 BCE."""

    # Process text
    doc = process_text(text, nlp)

    # Log entities, noun chunks, and sentences
    log_entities(doc)
    log_noun_chunks(doc)
    log_sentences(doc)

    entities = displacy.parse_ents(doc)
    dependencies = displacy.parse_deps(doc)
    spans = displacy.parse_spans(doc)

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    save_file(parse_entities(doc), f"{output_dir}/entities.json")
    save_file(parse_dependencies(doc), f"{output_dir}/dependencies.json")
    save_file(parse_sentences(doc), f"{output_dir}/sentences.json")
    save_file(parse_settings(doc), f"{output_dir}/settings.json")
    save_file(spans, f"{output_dir}/spans.json")

    # Visualize dependencies
    # displacy.serve(doc, style="dep", port=5002, options={"colors": {
    #     "PER": "#ff9999", "LOC": "#99ff99", "DATE": "#9999ff"}})


if __name__ == "__main__":
    main()
