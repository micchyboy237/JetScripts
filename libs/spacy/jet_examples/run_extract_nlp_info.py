from typing import List, Dict, Any
from jet.file.utils import save_file
import spacy
from spacy.tokens import Doc
import os
import shutil

class SpacyProcessor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize spaCy with the specified model."""
        self.nlp = spacy.load(model_name)

    def process_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process a list of texts and extract linguistic features."""
        results = []
        for text in texts:
            doc = self.nlp(text)
            result = {
                "text": text,
                "entities": self._extract_entities(doc),
                "sentences": self._extract_sentences(doc),
                "tokens": self._extract_tokens(doc),
                "pos_tags": self._extract_pos_tags(doc),
                "dependencies": self._extract_dependencies(doc),
                "noun_chunks": self._extract_noun_chunks(doc)
            }
            results.append(result)
        return results

    def _extract_entities(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract named entities from a spaCy Doc."""
        return [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]

    def _extract_sentences(self, doc: Doc) -> List[str]:
        """Extract sentences from a spaCy Doc."""
        return [sent.text.strip() for sent in doc.sents]

    def _extract_tokens(self, doc: Doc) -> List[str]:
        """Extract tokens from a spaCy Doc."""
        return [token.text for token in doc]

    def _extract_pos_tags(self, doc: Doc) -> List[Dict[str, str]]:
        """Extract part-of-speech tags from a spaCy Doc."""
        return [{"text": token.text, "pos": token.pos_, "tag": token.tag_} for token in doc]

    def _extract_dependencies(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract dependency parse information from a spaCy Doc."""
        return [{"text": token.text, "dep": token.dep_, "head": token.head.text} for token in doc]

    def _extract_noun_chunks(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract noun chunks from a spaCy Doc."""
        return [{"text": chunk.text, "root": chunk.root.text, "start": chunk.start_char, "end": chunk.end_char} for chunk in doc.noun_chunks]

    def save_results(self, results: List[Dict[str, Any]], output_dir: str) -> None:
        """Save extracted features to separate JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        
        feature_types = ["entities", "sentences", "tokens", "pos_tags", "dependencies", "noun_chunks"]
        
        for feature in feature_types:
            feature_data = {f"doc_{i}": result[feature] for i, result in enumerate(results)}
            output_path = os.path.join(output_dir, f"{feature}.json")
            save_file(feature_data, output_path)

def main():
    """Example usage of SpacyProcessor."""
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    processor = SpacyProcessor()
    sample_texts = [
        "Apple is launching a new iPhone in New York next week.",
        "Elon Musk founded SpaceX in 2002."
    ]
    results = processor.process_texts(sample_texts)
    processor.save_results(results, output_dir)

if __name__ == "__main__":
    main()