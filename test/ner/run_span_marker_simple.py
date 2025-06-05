from jet.logger import logger
from jet.transformers.formatters import format_json
import spacy
import numpy as np
from span_marker import SpanMarkerModel
from sklearn.manifold import TSNE
from typing import List, Dict, Tuple
import pytest

# Initialize models
nlp = spacy.load("en_core_web_sm")
model = SpanMarkerModel.from_pretrained(
    "tomaarsen/span-marker-bert-base-fewnerd-fine-super").to("mps")

text = "Apple is based in California."

doc = nlp(text)
spans = model.predict(doc)

entities = [
    {"text": span["text"], "label": span["label"]}
    for span in spans
]

logger.gray("\nSpans:")
logger.success(format_json(spans))
