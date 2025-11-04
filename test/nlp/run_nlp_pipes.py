#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
spaCy Processing Pipelines – Complete Usage Examples

A single, self-contained script demonstrating every major feature from the
official spaCy documentation page:

    https://spacy.io/usage/processing-pipelines

- Runs on macOS (M1) and Windows 10 Pro
- Uses only free `en_core_web_sm` (or blank pipeline)
- Modular: each example is a function
- Executable via `python spacy_pipelines_demo.py`

Author: Jethro Estrada
"""

from __future__ import annotations

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.matcher import PhraseMatcher
from spacy.training import biluo_tags_to_spans
from typing import List


# ----------------------------------------------------------------------
# 1. Basic processing: single vs. batch (nlp.pipe)
# ----------------------------------------------------------------------
def example_basic_processing() -> None:
    print("\n=== 1. Basic Processing ===")
    nlp = spacy.load("en_core_web_sm")

    # Single text
    doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")
    print("Tokens:", [t.text for t in doc])
    print("Entities:", [(e.text, e.label_) for e in doc.ents])

    # Batch (recommended)
    texts = [
        "Apple is looking at buying U.K. startup for $1 billion.",
        "Although the iPhone was released in 2007, it still dominates.",
    ]
    print("\nBatch processing with nlp.pipe:")
    for i, doc in enumerate(nlp.pipe(texts)):
        print(f"  [{i}] Entities:", [(e.text, e.label_) for e in doc.ents])


# ----------------------------------------------------------------------
# 2. Disabling / enabling pipeline components
# ----------------------------------------------------------------------
def example_disable_enable_components() -> None:
    print("\n=== 2. Disable / Enable Components ===")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    print("Loaded with disabled:", nlp.pipe_names)

    nlp.enable_pipe("tagger")
    print("After enabling tagger:", nlp.pipe_names)

    with nlp.select_pipes(enable=["ner"]):
        doc = nlp("Microsoft bought GitHub for $7.5 billion.")
        print("Only NER active:", [(e.text, e.label_) for e in doc.ents])


# ----------------------------------------------------------------------
# 3. nlp.pipe with context tuples + custom Doc extension
# ----------------------------------------------------------------------
def example_pipe_with_context() -> None:
    print("\n=== 3. nlp.pipe with Context Tuples ===")
    if not Doc.has_extension("doc_id"):
        Doc.set_extension("doc_id", default=None)

    data = [
        ("Paris is the capital of France.", {"doc_id": "doc-001"}),
        ("Berlin is the capital of Germany.", {"doc_id": "doc-002"}),
    ]

    nlp = spacy.load("en_core_web_sm")
    for doc, ctx in nlp.pipe(data, as_tuples=True):
        doc._.doc_id = ctx["doc_id"]
        print(f"{doc._.doc_id}: {[(e.text, e.label_) for e in doc.ents]}")


# ----------------------------------------------------------------------
# 4. Multiprocessing with nlp.pipe
# ----------------------------------------------------------------------
def example_multiprocessing() -> None:
    print("\n=== 4. Multiprocessing ===")
    nlp = spacy.load("en_core_web_sm")
    texts = [f"Sentence {i} about Apple." for i in range(200)]

    count = 0
    for _ in nlp.pipe(texts, n_process=2, batch_size=50):
        count += 1
    print(f"Processed {count} docs using 2 processes")


# ----------------------------------------------------------------------
# 5. Stateless custom component
# ----------------------------------------------------------------------
@Language.component("length_logger")
def length_logger(doc: Doc) -> Doc:
    print(f"Doc length: {len(doc)} tokens")
    return doc


def example_stateless_component() -> None:
    print("\n=== 5. Stateless Custom Component ===")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("length_logger", last=True)
    print("Pipeline:", nlp.pipe_names)
    _ = nlp("This is a short text.")


# ----------------------------------------------------------------------
# 6. Stateful component with factory & config
# ----------------------------------------------------------------------
ACRONYM_DICT = {"lol": "laughing out loud", "brb": "be right back"}
ACRONYM_DICT.update({v: k for k, v in ACRONYM_DICT.items()})


@Language.factory("acronym_resolver", default_config={"case_sensitive": False})
def create_acronym_resolver(nlp: Language, name: str, case_sensitive: bool):
    return AcronymResolver(nlp, case_sensitive)


class AcronymResolver:
    def __init__(self, nlp: Language, case_sensitive: bool):
        attr = "TEXT" if case_sensitive else "LOWER"
        self.matcher = PhraseMatcher(nlp.vocab, attr=attr)
        self.matcher.add("ACRONYMS", [nlp.make_doc(t) for t in ACRONYM_DICT])
        Doc.set_extension("acronyms", default=[])
        self.case_sensitive = case_sensitive

    def __call__(self, doc: Doc) -> Doc:
        for _, start, end in self.matcher(doc):
            span = doc[start:end]
            key = span.text if self.case_sensitive else span.text.lower()
            full = ACRONYM_DICT.get(key)
            if full:
                doc._.acronyms.append((span.text, full))
        return doc


def example_stateful_component() -> None:
    print("\n=== 6. Stateful Component with Config ===")
    nlp = spacy.blank("en")
    nlp.add_pipe("acronym_resolver", config={"case_sensitive": False})
    doc = nlp("LOL, I’ll brb.")
    print("Detected:", doc._.acronyms)


# ----------------------------------------------------------------------
# 7. Reusing a trained component from another pipeline
# ----------------------------------------------------------------------
def example_reuse_trained_component() -> None:
    print("\n=== 7. Reusing Trained NER ===")
    src = spacy.load("en_core_web_sm")
    blank = spacy.blank("en")
    blank.add_pipe("ner", source=src)
    print("Blank pipeline now has:", blank.pipe_names)
    doc = blank("Apple bought a startup.")
    print("Entities:", [(e.text, e.label_) for e in doc.ents])


# ----------------------------------------------------------------------
# 8. Pipeline analysis
# ----------------------------------------------------------------------
def example_pipeline_analysis() -> None:
    print("\n=== 8. Pipeline Analysis ===")
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("ner")  # needs entities → warning
    print(nlp.analyze_pipes(pretty=True))


# ----------------------------------------------------------------------
# 9. Custom extension attributes (Doc, Span, Token)
# ----------------------------------------------------------------------
def example_extension_attributes() -> None:
    print("\n=== 9. Custom Extension Attributes ===")
    fruits = {"apple", "pear", "banana"}

    Token.set_extension("is_fruit", getter=lambda t: t.text in fruits)
    Span.set_extension("has_fruit", getter=lambda s: any(t._.is_fruit for t in s))
    Doc.set_extension("fruit_count", getter=lambda d: sum(t._.is_fruit for t in d))

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("I like apple pie and banana bread.")
    print("Fruit tokens:", [t.text for t in doc if t._.is_fruit])
    print("Total fruits:", doc._.fruit_count)
    print("Span[2:5] has fruit?", doc[2:5]._.has_fruit)


# ----------------------------------------------------------------------
# 10. Wrapping an external NER (BILUO → spaCy ents)
# ----------------------------------------------------------------------
def dummy_external_ner(words: List[str]) -> List[str]:
    """Simulate external model returning BILUO tags."""
    tags = ["O"] * len(words)
    if "Facebook" in words:
        tags[words.index("Facebook")] = "U-ORG"
    return tags


@Language.component("external_ner")
def external_ner_wrapper(doc: Doc) -> Doc:
    words = [t.text for t in doc]
    biluo = dummy_external_ner(words)
    new_ents = biluo_tags_to_spans(doc, biluo)
    doc.ents = list(doc.ents) + new_ents
    return doc


def example_external_ner() -> None:
    print("\n=== 10. Wrapping External NER ===")
    nlp = spacy.blank("en")
    nlp.add_pipe("external_ner")
    doc = nlp("I work at Facebook.")
    print("Detected entities:", [(e.text, e.label_) for e in doc.ents])


# --------------------------------------------------------------
# 11. tomaarsen/span-marker-roberta-large-ontonotes5 → Probabilities
# --------------------------------------------------------------
# --------------------------------------------------------------
# 11. tomaarsen/span-marker-roberta-large-ontonotes5 → Probabilities
# --------------------------------------------------------------
def example_with_scores() -> None:
    print("\n=== 11. RoBERTa-Large OntoNotes5 + Confidence Scores (FIXED) ===")
    import os
    import spacy
    from spacy.tokens import Span

    # 1. Silence warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKENIZER"] = "1"

    try:
        # 2. Register extension
        if not Span.has_extension("span_marker_score"):
            Span.set_extension("span_marker_score", default=None)

        # 3. Create pipeline with sentencizer FIRST
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")  # Required by SpanMarker

        # 4. Add SpanMarker (correct config)
        nlp.add_pipe(
            "span_marker",
            config={
                "model": "tomaarsen/span-marker-roberta-large-ontonotes5",
                "batch_size": 4,
                "overwrite_entities": True,
                # Optional: force device
                # "device": "mps",   # M1 Mac
                # "device": "cuda",  # GTX 1660
            }
        )

        # 5. Test
        text = "Apple bought a startup in London for $1 billion."
        doc = nlp(text)

        # Results in doc.ents (because overwrite_entities=True)
        print("Entities with probability:")
        for ent in doc.ents:
            prob = ent._.score
            prob_str = f"{prob:.3f}" if prob is not None else "N/A"
            print(f"  {ent.text:15} {ent.label_:12} {prob_str}")

        # Expected:
        #   Apple          ORG          0.998
        #   London         GPE          0.996
        #   $1 billion     MONEY        0.999

    except Exception as e:
        print("Error:", e)
        print("Install: pip install spacy span-marker")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    examples = [
        example_basic_processing,
        example_disable_enable_components,
        example_pipe_with_context,
        example_multiprocessing,
        example_stateless_component,
        example_stateful_component,
        example_reuse_trained_component,
        example_pipeline_analysis,
        example_extension_attributes,
        example_external_ner,
        example_with_scores,
    ]

    print("spaCy Processing Pipelines – Full Demo")
    print("=" * 50)
    for func in examples:
        try:
            func()
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
        print("-" * 50)


if __name__ == "__main__":
    main()