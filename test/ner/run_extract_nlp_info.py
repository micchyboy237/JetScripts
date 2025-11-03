import os
import shutil
import spacy
from typing import Dict, List
from jet.code.html_utils import convert_dl_blocks_to_md
from jet.file.utils import load_file, save_file
from jet.logger import logger
from span_marker import SpanMarkerModel
from jet.scrapers.header_hierarchy import HtmlHeaderDoc, extract_header_hierarchy
from jet.wordnet.text_chunker import chunk_texts_with_data
from tqdm import tqdm
from pydantic import BaseModel
from spacy.tokens import Token
from dataclasses import dataclass
from spacy.tokens import Doc


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

model = SpanMarkerModel.from_pretrained(
    "tomaarsen/span-marker-roberta-large-ontonotes5",
).to("mps")

# Load the spaCy model
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("span_marker", config={
    "model": "tomaarsen/span-marker-roberta-large-ontonotes5",
    "device": "mps",
})

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
class DocPOSMeta:
    is_digit: bool
    is_currency: bool
    like_email: bool
    like_num: bool
    like_url: bool


@dataclass
class DocPOS:
    """Represents a single token's Part-of-Speech details with sentence context."""
    text: str
    lemma: str
    pos: str          # Coarse-grained POS (NOUN, VERB, ADJ, etc.)
    tag: str          # Fine-grained tag (NNP, VBD, JJ, etc.)
    morph: str        # Morphological features
    dep: str          # Syntactic dependency
    ent: str          # Named entity label
    head_text: str    # Head token text
    lang: str         # Language code for the token (e.g. 'en')
    sentence_text: str  # Full sentence context
    meta: DocPOSMeta  # Miscellaneous token metadata flags (digit, currency, etc.)


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


def process_predictions(text: str, nlp: spacy.language.Language, model: SpanMarkerModel) -> List[SpanMarkerWord]:
    """Process predictions with SpanMarker model, returning SpanMarkerWord predictions."""
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
    return processed_predictions


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


def parse_pos(doc: Doc) -> List[DocPOS]:
    """Parse a spaCy Doc into a list of DocPOS objects with POS, morph, dep, and sentence context."""
    # Pre-compute sentence boundaries for fast lookup
    sent_starts: Dict[int, str] = {sent.start: sent.text for sent in doc.sents}

    def _sentence_for_token(token_idx: int) -> str:
        """Get sentence text for token using binary-search-like lookup."""
        for start in sorted(sent_starts.keys(), reverse=True):
            if start <= token_idx:
                return sent_starts[start]
        return ""

    return [
        DocPOS(
            text=token.text,
            lemma=token.lemma_,
            pos=token.pos_,
            tag=token.tag_,
            morph=token.morph.to_dict() if token.morph else {},
            dep=token.dep_,
            ent=token.ent_type_,
            head_text=token.head.text,
            lang=token.lang_,
            sentence_text=_sentence_for_token(token.i),
            meta=DocPOSMeta(
                is_digit=token.is_digit,
                is_currency=token.is_currency,
                like_email=token.like_email,
                like_num=token.like_num,
                like_url=token.like_url,
            )
        )
        for token in doc
        if not token.is_space and not token.is_punct  # Filter whitespace/punctuation
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


def parse_noun_chunks(doc: Doc) -> List[DocNounChunk]:
    """Parse a spaCy Doc into a list of DocNounChunk objects with full sentence context.

    Uses pre-computed sentence start indices for O(log n) lookup per token.
    Returns generic, reusable objects suitable for serialization or further analysis.
    """
    # Pre-compute sentence boundaries for fast lookup
    sent_starts: Dict[int, str] = {sent.start: sent.text for sent in doc.sents}

    def _sentence_for_token(token: Token) -> str:
        # Binary search equivalent using sorted keys in reverse
        for start in sorted(sent_starts.keys(), reverse=True):
            if start <= token.i:
                return sent_starts[start]
        return ""

    return [
        DocNounChunk(
            text=chunk.text,
            root_text=chunk.root.text,
            root_dep=chunk.root.dep_,
            root_head_text=chunk.root.head.text,
            context=_sentence_for_token(chunk.root)
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


def extract_nlp(text: str) -> dict:
    logger.info("Running spacy NER extraction")
    doc = nlp(text)
    spacy_entities = doc.ents

    logger.info("Running span marker NER extraction")
    predictions = process_predictions(text, nlp, model)
    entities = parse_entities(doc, predictions)
    pos = parse_pos(doc)
    dependencies = parse_dependencies(doc)
    noun_chunks = parse_noun_chunks(doc)
    sentences = parse_sentences(doc)

    return {
        "pos": pos,
        "entities": entities,
        "dependencies": dependencies,
        "noun_chunks": noun_chunks,
        "sentences": sentences,
        "spacy_entities": spacy_entities,
        # "dependencies": dependencies,
        # "constituencies": constituencies,
        # "scenes": scenes,
        # "sentence_details": sentence_details,
    }

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"

    html_str: str = load_file(html_file)
    html_str = convert_dl_blocks_to_md(html_str)
    save_file(html_str, f"{OUTPUT_DIR}/page.html")

    headings: List[HtmlHeaderDoc] = extract_header_hierarchy(html_str)
    save_file(headings, f"{OUTPUT_DIR}/headings.json")

    doc_ids = [h["id"] for h in headings]
    contents = [h["content"] for h in headings]

    chunks_with_data = chunk_texts_with_data(
        contents,
        ids=doc_ids,
        chunk_size=512,
        chunk_overlap=50,
        model="qwen3-instruct-2507:4b",
    )

    for chunk_idx, chunk in enumerate(tqdm(chunks_with_data, desc="Processing headings...")):
        if not chunk:
            continue

        header_idx = doc_ids.index(chunk["doc_id"])
        sub_output_dir = f"{OUTPUT_DIR}/heading_{header_idx + 1}"
        sub_output_dir = f"{sub_output_dir}_{chunk_idx + 1}"
    
        save_file({
            "id": chunk["doc_id"],
            "doc_id": chunk["doc_id"],
            "doc_index": chunk["doc_index"],
            "chunk_index": chunk["chunk_index"],
            "num_tokens": chunk["num_tokens"],
        }, f"{sub_output_dir}/meta.json")
        save_file(chunk["content"], f"{sub_output_dir}/chunk.txt")

        results = extract_nlp(chunk["content"])
        for key, nlp_results in results.items():
            save_file(nlp_results, f"{sub_output_dir}/{key}.json")
