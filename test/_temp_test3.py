# entity_extractor.py
from span_marker import SpanMarkerModel
from typing import List, Dict


class EntityExtractor:
    def __init__(self, model_name: str = "tomaarsen/span-marker-mbert-base-multinerd"):
        """
        Initialize the SpanMarker model for NER.
        """
        self.model = SpanMarkerModel.from_pretrained(model_name)

    def extract_entities(self, text: str, threshold: float = 0.7) -> List[Dict[str, any]]:
        """
        Extract entities from text with confidence scores above the threshold.

        Args:
            text (str): Input text for entity extraction.
            threshold (float): Minimum confidence score for entities (default: 0.7).

        Returns:
            List[Dict[str, any]]: List of entities with span, label, and score.
        """
        entities = self.model.predict(text)
        return [
            {"span": e["span"], "label": e["label"], "score": e["score"]}
            for e in entities
            if e["score"] >= threshold
        ]


extractor = EntityExtractor()
text = "Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris."
entities = extractor.extract_entities(text, threshold=0.7)
for entity in entities:
    print(
        f"Span: {entity['span']}, Label: {entity['label']}, Score: {entity['score']}")
