from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
import spacy
import nltk
from nltk.tokenize import sent_tokenize

nlp = spacy.load("en_core_web_md")


def is_heading(line: str) -> bool:
    """Heuristically determine if a line is a heading."""
    stripped = line.strip()
    return (
        stripped.isupper() or
        stripped.istitle() and len(stripped.split()) < 10 or
        len(stripped) < 50 and not stripped.endswith(".")
    )


def split_paragraphs(text: str) -> list:
    """Split raw text into paragraphs."""
    return [p.strip() for p in text.strip().split('\n') if p.strip()]


def group_sentences_by_similarity(paragraphs: list, threshold=0.85) -> list:
    """Group sentences based on semantic similarity."""
    segments = []
    current_segment = []

    for paragraph in paragraphs:
        if is_heading(paragraph):
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            segments.append([f"[HEADING] {paragraph}"])
        else:
            sentences = sent_tokenize(paragraph)
            for sentence in sentences:
                doc = nlp(sentence)
                if not current_segment:
                    current_segment.append(sentence)
                else:
                    prev_doc = nlp(current_segment[-1])
                    if doc.similarity(prev_doc) > threshold:
                        current_segment.append(sentence)
                    else:
                        segments.append(current_segment)
                        current_segment = [sentence]

    if current_segment:
        segments.append(current_segment)

    return segments


sample_text = """
PRODUCT OVERVIEW
Our new model introduces several smart features designed to improve usability and performance.
It includes advanced AI algorithms and a new camera sensor.

TECHNICAL SPECIFICATIONS
The processor is a 2.6GHz octa-core chip.
It supports 5G connectivity and has 12GB of RAM.
Battery capacity is rated at 4500mAh.

USER FEEDBACK
Many users praised the battery life and overall speed.
Some users encountered issues with Bluetooth connectivity.
"""

segments = group_sentences_by_similarity(sample_text.split('\n'))

copy_to_clipboard(segments)
logger.success(format_json(segments))

for idx, seg in enumerate(segments):
    print(f"\n--- Segment {idx + 1} ---")
    for line in seg:
        print(line)
