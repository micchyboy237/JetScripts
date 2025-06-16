from jet.logger import logger
from jet.transformers.formatters import format_json
import spacy
import numpy as np
from span_marker import SpanMarkerModel
from sklearn.manifold import TSNE
from typing import List, Dict, Tuple
from pydantic import BaseModel

# Define SpanMarkerWord model


class SpanMarkerWord(BaseModel):
    text: str
    lemma: str
    start_idx: int
    end_idx: int
    score: float
    label: str

    def __str__(self) -> str:
        return self.text


# Initialize models
nlp = spacy.load("en_core_web_sm")
model = SpanMarkerModel.from_pretrained(
    "tomaarsen/span-marker-bert-base-fewnerd-fine-super").to("cpu")

# Input text
text = "Apple is based in California."

# Process document and predictions
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

# Log entities
logger.newline()
logger.debug(f"Extracted Entities ({len(processed_predictions)}):")
for entity in processed_predictions:
    logger.newline()
    logger.log("Text:", entity.text, colors=["WHITE", "INFO"])
    logger.log("Lemma:", entity.lemma, colors=["WHITE", "INFO"])
    logger.log("Label:", entity.label, colors=["WHITE", "INFO"])
    logger.log("Start:", f"{entity.start_idx}", colors=["WHITE", "SUCCESS"])
    logger.log("End:", f"{entity.end_idx}", colors=["WHITE", "SUCCESS"])
    logger.log("Score:", f"{entity.score:.4f}", colors=["WHITE", "SUCCESS"])
    logger.log("---")

# Log noun chunks
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

# Log sentences
logger.newline()
logger.debug(f"Extracted Sentences ({len(list(doc.sents))}):")
for i, sent in enumerate(doc.sents, 1):
    logger.newline()
    logger.log(f"Sentence {i}:", sent.text, colors=["WHITE", "INFO"])
    logger.log("Start Char:", f"{sent.start_char}",
               colors=["WHITE", "SUCCESS"])
    logger.log("End Char:", f"{sent.end_char}", colors=["WHITE", "SUCCESS"])
    logger.log("Token Count:", f"{len(sent)}", colors=["WHITE", "SUCCESS"])
    logger.log("---")

# Log spans in simplified format
spans = [
    {"text": span.text, "label": span.label}
    for span in processed_predictions
]
logger.gray("\nSpans:")
logger.success(format_json(spans))
