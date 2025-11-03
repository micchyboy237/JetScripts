from jet.file.utils import save_file
from jet.logger import logger
from pydantic import BaseModel
from spacy.tokens import SpanGroup
from dataclasses import dataclass
from spacy.tokens import Doc, Span
import spacy
from spacy import displacy
from span_marker import SpanMarkerModel
from typing import List, Optional
import os

# SpanMarkerWord with label


class SpanMarkerWord(BaseModel):
    text: str
    lemma: str
    start_idx: int
    end_idx: int
    score: float
    label: str

    def __str__(self) -> str:
        return self.text


@dataclass
class DocSentence:
    text: str
    start_char: int
    end_char: int
    token_count: int


@dataclass
class DocEntity:
    text: str
    lemma: str  # Added lemma field
    label: str
    start_char: int
    end_char: int
    score: float
    vector_norm: float | None


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


def process_text(text: str, nlp: spacy.language.Language, model: SpanMarkerModel) -> tuple[Doc, List[SpanMarkerWord]]:
    """Process text with spaCy pipeline and SpanMarker model, returning SpanMarkerWord predictions."""
    doc = nlp(text)
    predictions = model.predict(text)
    processed_predictions = [
        SpanMarkerWord(
            text=pred["span"],
            lemma=nlp(pred["span"])[0].lemma_ if pred["span"] else "",
            start_idx=pred["char_start_index"],
            end_idx=pred["char_end_index"],
            score=pred["score"],
            label=pred["label"]
        )
        for pred in predictions
    ]
    return doc, processed_predictions


def log_entities(predictions: List[SpanMarkerWord]) -> None:
    """Log named entities with relevant details."""
    logger.newline()
    logger.debug(f"Extracted Entities ({len(predictions)}):")
    for entity in predictions:
        logger.newline()
        logger.log("Text:", entity.text, colors=["WHITE", "INFO"])
        logger.log("Lemma:", entity.lemma, colors=["WHITE", "INFO"])
        logger.log("Label:", entity.label, colors=["WHITE", "INFO"])
        logger.log("Start:", f"{entity.start_idx}",
                   colors=["WHITE", "SUCCESS"])
        logger.log("End:", f"{entity.end_idx}", colors=["WHITE", "SUCCESS"])
        logger.log("Score:", f"{entity.score:.4f}",
                   colors=["WHITE", "SUCCESS"])
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


def parse_entities(doc: Doc, predictions: List[SpanMarkerWord]) -> List[DocEntity]:
    """Parse SpanMarkerWord predictions into a list of DocEntity objects."""
    return [
        DocEntity(
            text=entity.text,
            lemma=entity.lemma,  # Include lemma
            label=entity.label,
            start_char=entity.start_idx,
            end_char=entity.end_idx,
            score=entity.score,
            vector_norm=(
                doc[entity.start_idx:entity.end_idx].vector_norm
                if doc[entity.start_idx:entity.end_idx].has_vector
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


def create_span_group(doc: Doc, predictions: List[SpanMarkerWord]) -> SpanGroup:
    """Create a SpanGroup from SpanMarkerWord predictions for visualization."""
    spans = []
    for entity in predictions:
        start_token, end_token = char_to_token_index(
            doc, entity.start_idx, entity.end_idx)
        if start_token is not None and end_token is not None and start_token < len(doc) and end_token <= len(doc):
            try:
                span = Span(
                    doc,
                    start_token,
                    end_token,
                    label=entity.label,
                    kb_id=f"score:{entity.score:.4f}"
                )
                spans.append(span)
            except IndexError as e:
                logger.error(f"Error creating span for entity '{entity.text}' "
                             f"(char {entity.start_idx}:{entity.end_idx}): {e}")
        else:
            logger.warning(f"Skipping entity '{entity.text}' due to invalid token indices "
                           f"(char {entity.start_idx}:{entity.end_idx})")
    return SpanGroup(doc, name="entities", spans=spans)


def main():
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    model = SpanMarkerModel.from_pretrained(
        "tomaarsen/span-marker-mbert-base-multinerd").to("cpu")

    # Input text
    text = """Title: Headhunted to Another World: From Salaryman to Big Four!
Isekai
Fantasy
Comedy
Release Date : January 1, 2025
Release Date
: January 1, 2025
Japanese Title : Salaryman ga Isekai ni Ittara Shitennou ni Natta Hanashi Studio : Geek Toys, CompTown Based On : Manga Creator : Benigashira Streaming Service(s) : Crunchyroll
Japanese Title
: Salaryman ga Isekai ni Ittara Shitennou ni Natta Hanashi

Studio
: Geek Toys, CompTown

Based On
: Manga

Creator
: Benigashira

Streaming Service(s)
: Crunchyroll
Powered by
Expand Collapse
Plenty of 2025 isekai anime will feature OP protagonists capable of brute-forcing their way through any and every encounter, so it is always refreshing when an MC comes along that relies on brain rather than brawn. A competent office worker who feels underappreciated, Uchimura is suddenly summoned to another world by a demonic ruler, who comes with quite an unusual offer: Join the crew as one of the Heavenly Kings. So, Uchimura starts a new career path that tasks him with tackling challenges using his expertise in discourse and sales.
Related"""

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
    displacy.serve(doc, style="dep", port=5002)


if __name__ == "__main__":
    main()
