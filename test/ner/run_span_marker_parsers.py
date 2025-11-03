from jet.file.utils import save_file
from jet.logger import logger
from pydantic import BaseModel
from spacy.tokens import SpanGroup, Token
from dataclasses import dataclass
from spacy.tokens import Doc, Span
import spacy
from spacy import displacy
from span_marker import SpanMarkerModel
from typing import List, Optional
import os
from typing import Dict, Any

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
    context: str


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
    """Parse a spaCy Doc into a list of DocNounChunk objects containing noun chunk details
    plus the full sentence context."""
    # Pre-compute sentence boundaries for fast lookup
    sent_starts = {sent.start: sent.text for sent in doc.sents}

    def _sentence_for_token(token: Token) -> str:
        # Find the sentence start index that is <= token.i
        for start in reversed(sorted(sent_starts.keys())):
            if start <= token.i:
                return sent_starts[start]
        return ""

    return [
        DocNounChunk(
            text=chunk.text,
            root_text=chunk.root.text,
            root_dep=chunk.root.dep_,
            root_head_text=chunk.root.head.text,
            context=_sentence_for_token(chunk.root)  # full sentence text
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


def create_noun_chunk_spans(doc: Doc) -> None:
    """Assign noun chunks to a custom SpanGroup for visualization."""
    spans = [
        Span(doc, chunk.start, chunk.end, label="NOUN_CHUNK")
        for chunk in doc.noun_chunks
    ]
    doc.spans["noun_chunks"] = spans


def create_sentence_spans(doc: Doc) -> None:
    """Assign sentences to a custom SpanGroup for visualization."""
    spans = [
        Span(doc, sent.start, sent.end, label="SENTENCE")
        for sent in doc.sents
    ]
    doc.spans["sentences"] = spans


def create_custom_spans(doc: Doc) -> None:
    """Assign arbitrary custom spans (e.g., key phrases) to a SpanGroup."""
    # Example: Highlight specific phrases
    spans = [
        doc[:3],  # First three tokens
        doc[5:8],  # Tokens 5-8
    ]
    for span in spans:
        span.label_ = "CUSTOM_PHRASE"
    doc.spans["custom_phrases"] = spans


def create_span_group_from_predictions(doc: Doc, predictions: List[SpanMarkerWord], key: str = "sc") -> None:
    """Create and assign a SpanGroup from predictions to a specified doc.spans key."""
    spans = []
    for entity in predictions:
        start_token, end_token = char_to_token_index(doc, entity.start_idx, entity.end_idx)
        if start_token is not None and end_token is not None:
            try:
                span = Span(doc, start_token, end_token, label=entity.label)
                spans.append(span)
            except IndexError:
                logger.warning(f"Invalid span for '{entity.text}'")
    doc.spans[key] = SpanGroup(doc, name=key, spans=spans)


def parse_spans_entities(doc: Doc):
    """Parse spans from a custom 'entities' key"""
    parsed_span_entities = displacy.parse_spans(doc, options={"spans_key": "entities"})
    text = parsed_span_entities["text"]
    spans = parsed_span_entities["spans"]
    result = {
        "text": text,
        "spans": [{
            **span,
            "span": text[span["start"] : span["end"]]
        } for span in spans]
    }
    return result


def parse_spans_noun_chunks(doc: Doc) -> Dict[str, Any]:
    """Parse spans from a 'noun_chunks' key."""
    create_noun_chunk_spans(doc)
    parsed = displacy.parse_spans(doc, options={"spans_key": "noun_chunks"})
    text = parsed["text"]
    spans = parsed["spans"]
    result = {
        "text": text,
        "spans": [
            {**span, "span": text[span["start"]: span["end"]]}
            for span in spans
        ]
    }
    return result


def parse_spans_sentences(doc: Doc) -> Dict[str, Any]:
    """Parse spans from a 'sentences' key."""
    create_sentence_spans(doc)
    parsed = displacy.parse_spans(doc, options={"spans_key": "sentences"})
    text = parsed["text"]
    spans = parsed["spans"]
    result = {
        "text": text,
        "spans": [
            {**span, "span": text[span["start"]: span["end"]]}
            for span in spans
        ]
    }
    return result


def parse_spans_custom_phrases(doc: Doc) -> Dict[str, Any]:
    """Parse spans from a 'custom_phrases' key."""
    create_custom_spans(doc)
    parsed = displacy.parse_spans(doc, options={"spans_key": "custom_phrases"})
    text = parsed["text"]
    spans = parsed["spans"]
    result = {
        "text": text,
        "spans": [
            {**span, "span": text[span["start"]: span["end"]]}
            for span in spans
        ]
    }
    return result


def parse_spans_ruler(doc: Doc) -> Dict[str, Any]:
    """Parse spans from the 'ruler' key (default for SpanRuler component)."""
    # Assume a SpanRuler has been added to the pipeline
    parsed = displacy.parse_spans(doc, options={"spans_key": "ruler"})
    text = parsed["text"]
    spans = parsed["spans"]
    result = {
        "text": text,
        "spans": [
            {**span, "span": text[span["start"]: span["end"]]}
            for span in spans
        ]
    }
    return result


def main():
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    model = SpanMarkerModel.from_pretrained(
        # "tomaarsen/span-marker-mbert-base-multinerd"
        "tomaarsen/span-marker-roberta-large-ontonotes5"
    ).to("mps")

    # Input text
    text = """Title: Headhunted to Another World: From Salaryman to Big Four!
Isekai
Fantasy
Comedy
Release Date: January 1, 2025
Japanese Title: Salaryman ga Isekai ni Ittara Shitennou ni Natta Hanashi Studio

Studio: Geek Toys, CompTown

Based On: Manga

Creator: Benigashira

Streaming Service(s): Crunchyroll
Powered by
Expand Collapse
Plenty of 2025 isekai anime will feature OP protagonists capable of brute-forcing their way through any and every encounter, so it is always refreshing when an MC comes along that relies on brain rather than brawn. A competent office worker who feels underappreciated, Uchimura is suddenly summoned to another world by a demonic ruler, who comes with quite an unusual offer: Join the crew as one of the Heavenly Kings. So, Uchimura starts a new career path that tasks him with tackling challenges using his expertise in discourse and sales.
Related"""

    # Process text (first pass: base pipeline + SpanMarker)
    doc, predictions = process_text(text, nlp, model)

    # Log entities, noun chunks, and sentences (from first pass)
    log_entities(predictions)
    log_noun_chunks(doc)
    log_sentences(doc)

    # Create reusable span groups from predictions
    create_span_group_from_predictions(doc, predictions, key="entities")

    # Add SpanRuler if not present
    if "span_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("span_ruler")
        ruler.add_patterns([{"label": "MOCK_RULER", "pattern": "Crunchyroll"}])

    # Re-process text to apply ruler and refresh noun_chunks/sents (pipeline-dependent)
    doc = nlp(text)
    # Re-apply SpanMarker predictions to refreshed doc
    _, predictions = process_text(text, nlp, model)  # Re-run for char indices alignment
    create_span_group_from_predictions(doc, predictions, key="entities")

    # Refresh other example spans on the final doc
    create_noun_chunk_spans(doc)
    create_sentence_spans(doc)
    create_custom_spans(doc)

    # Parse and save data
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    os.makedirs(output_dir, exist_ok=True)

    save_file([e.__dict__ for e in parse_entities(doc, predictions)],
              f"{output_dir}/entities.json")
    save_file([d.__dict__ for d in parse_dependencies(doc)],
              f"{output_dir}/dependencies.json")
    save_file([s.__dict__ for s in parse_sentences(doc)],
              f"{output_dir}/sentences.json")
    save_file(parse_settings(doc).__dict__, f"{output_dir}/settings.json")
    save_file(parse_spans_entities(doc), f"{output_dir}/span_entities.json")
    save_file(parse_spans_noun_chunks(doc), f"{output_dir}/span_noun_chunks.json")
    save_file(parse_spans_sentences(doc), f"{output_dir}/span_sentences.json")
    save_file(parse_spans_custom_phrases(doc), f"{output_dir}/span_custom_phrases.json")
    save_file(parse_spans_ruler(doc), f"{output_dir}/span_ruler.json")

    # Visualize dependencies
    displacy.serve(doc, style="dep", port=5002)


if __name__ == "__main__":
    main()
