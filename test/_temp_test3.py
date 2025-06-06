import shutil
from jet.file.utils import load_file, save_file
from jet.logger import logger
from spacy.tokens import SpanGroup
from dataclasses import dataclass
from spacy.tokens import Doc, Span
import spacy
from spacy import displacy
from span_marker import SpanMarkerModel
from typing import Dict, List, Optional
import os


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
    score: float  # Changed to float
    vector_norm: float | None  # Changed to float or None for cases with no vector


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


def process_text(text: str, nlp: spacy.language.Language, model: SpanMarkerModel) -> tuple[Doc, List[Dict]]:
    """Process text with spaCy pipeline and SpanMarker model."""
    doc = nlp(text)
    predictions = model.predict(text)
    return doc, predictions


def log_entities(predictions: List[Dict]) -> None:
    """Log named entities with relevant details."""
    logger.newline()
    logger.debug(f"Extracted Entities ({len(predictions)}):")
    for entity in predictions:
        logger.newline()
        logger.log("Text:", entity["span"], colors=["WHITE", "INFO"])
        logger.log("Label:", entity["label"], colors=["WHITE", "INFO"])
        logger.log("Start:", f"{entity['char_start_index']}", colors=[
                   "WHITE", "SUCCESS"])
        logger.log("End:", f"{entity['char_end_index']}", colors=[
                   "WHITE", "SUCCESS"])
        logger.log("Score:", f"{entity['score']:.4f}", colors=[
                   "WHITE", "SUCCESS"])
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


def parse_entities(doc: Doc, predictions: List[Dict]) -> List[DocEntity]:
    """Parse SpanMarker predictions into a list of DocEntity objects."""
    return [
        DocEntity(
            text=entity["span"],
            label=entity["label"],
            start_char=entity["char_start_index"],
            end_char=entity["char_end_index"],
            score=entity["score"],  # Keep as float
            vector_norm=(
                doc[entity["char_start_index"]:entity["char_end_index"]].vector_norm
                if doc[entity["char_start_index"]:entity["char_end_index"]].has_vector
                else None
            )
        )
        for entity in predictions
    ]


def parse_dependencies(doc: Doc) -> List[DocNounChunk]:
    """Parse a spaCy Doc into a list of DocNounChunk objects containing noun chunk details."""
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
    """Parse a spaCy Doc into a list of DocSentence objects containing sentence details."""
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
    """Parse a spaCy Doc's settings into a DocSettings object."""
    return DocSettings(
        lang=doc.lang_,
        direction=doc.vocab.writing_system.get("direction", "ltr")
    )


def char_to_token_index(doc: Doc, char_start: int, char_end: int) -> tuple[Optional[int], Optional[int]]:
    """Convert character indices to token indices in a spaCy Doc."""
    start_token = None
    end_token = None
    for token in doc:
        if token.idx <= char_start < token.idx + len(token.text):
            start_token = token.i
        if token.idx < char_end <= token.idx + len(token.text):
            end_token = token.i + 1
            break
    return start_token, end_token


def create_span_group(doc: Doc, predictions: List[Dict]) -> SpanGroup:
    """Create a SpanGroup from SpanMarker predictions for visualization."""
    spans = []
    for entity in predictions:
        start_token, end_token = char_to_token_index(
            doc, entity["char_start_index"], entity["char_end_index"])
        if start_token is not None and end_token is not None and start_token < len(doc) and end_token <= len(doc):
            try:
                span = Span(
                    doc,
                    start_token,
                    end_token,
                    label=entity["label"],
                    kb_id=f"score:{entity['score']:.4f}"
                )
                spans.append(span)
            except IndexError as e:
                logger.error(f"Error creating span for entity '{entity['span']}' "
                             f"(char {entity['char_start_index']}:{entity['char_end_index']}): {e}")
        else:
            logger.warning(f"Skipping entity '{entity['span']}' due to invalid token indices "
                           f"(char {entity['char_start_index']}:{entity['char_end_index']})")
    return SpanGroup(doc, name="entities", spans=spans)


def main():
    # Load documents
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir)

    docs = load_file(docs_file)

    # Load spaCy model
    nlp = spacy.load("en_core_web_md")

    # Add SpanMarker without spans_key (it defaults to "sc")
    nlp.add_pipe("span_marker", config={
        "model": "tomaarsen/span-marker-mbert-base-multinerd",
        "batch_size": 4,
        "device": "mps",
        "overwrite_entities": False
    }, last=True)

    # Input text
    text = docs[0]["text"]

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    model = SpanMarkerModel.from_pretrained(
        # "tomaarsen/span-marker-mbert-base-multinerd",
        "tomaarsen/span-marker-bert-base-fewnerd-fine-super",
    ).to("mps")

    # Process text
    doc, predictions = process_text(text, nlp, model)

    # Log entities, noun chunks, and sentences
    log_entities(predictions)
    log_noun_chunks(doc)
    log_sentences(doc)

    # Create span group for visualization
    doc.spans["entities"] = create_span_group(doc, predictions)

    # Parse and save data
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    save_file([e.__dict__ for e in parse_entities(
        doc, predictions)], f"{output_dir}/entities.json")
    save_file([d.__dict__ for d in parse_dependencies(doc)],
              f"{output_dir}/dependencies.json")
    save_file([s.__dict__ for s in parse_sentences(doc)],
              f"{output_dir}/sentences.json")
    save_file(parse_settings(doc).__dict__, f"{output_dir}/settings.json")
    save_file(displacy.parse_spans(doc, options={
              "spans_key": "entities"}), f"{output_dir}/spans.json")

    # Visualize spans
    options = {
        "spans_key": "entities",
        "colors": {"PER": "#ff9999", "LOC": "#99ff99", "DATE": "#9999ff"}
    }
    displacy.render(doc, style="span", options=options, jupyter=False)


if __name__ == "__main__":
    main()
