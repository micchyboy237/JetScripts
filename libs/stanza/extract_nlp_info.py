import os
import shutil
import stanza
from typing import List
from pathlib import Path

DEFAULT_MODEL_DIR = os.getenv(
    'STANZA_RESOURCES_DIR',
    os.path.join(os.path.expanduser("~/.cache"), "stanza_resources")
)

# Sample text for processing
SAMPLE_TEXT = (
    "Barack Obama was born in Hawaii. He was the president. "
    "The White House is in Washington, D.C."
)

def initialize_pipeline(processors: str, lang: str = "en") -> stanza.Pipeline:
    """Initialize Stanza pipeline with specified processors."""
    return stanza.Pipeline(lang=lang, dir=DEFAULT_MODEL_DIR, processors=processors, use_gpu=True)

def tokenize_example() -> str:
    """Demonstrate tokenization and sentence segmentation."""
    nlp = initialize_pipeline("tokenize")
    doc = nlp(SAMPLE_TEXT)
    result = ["Tokenization Example:"]
    for sent in doc.sentences:
        tokens = [token.text for token in sent.tokens]
        result.append(f"Sentence: {' '.join(tokens)}")
        result.append(f"Tokens: {tokens}")
    return result

def mwt_example() -> str:
    """Demonstrate multi-word token expansion for English."""
    nlp = initialize_pipeline("tokenize,mwt", lang="en")
    doc = nlp("I don't like to swim.")  # English example with MWT (don't -> do + not)
    result = ["Multi-Word Token Expansion Example (English):"]
    for sent in doc.sentences:
        tokens = [token.text for token in sent.tokens]
        words = [word.text for word in sent.words]
        result.append(f"Tokens: {tokens}")
        result.append(f"Words: {words}")
    return result

def pos_example() -> str:
    """Demonstrate part-of-speech tagging."""
    nlp = initialize_pipeline("tokenize,mwt,pos")
    doc = nlp(SAMPLE_TEXT)
    result = ["Part-of-Speech Tagging Example:"]
    for sent in doc.sentences:
        pos_tags = [(word.text, word.upos, word.xpos, word.feats) for word in sent.words]
        result.append(f"Sentence: {' '.join(word.text for word in sent.words)}")
        result.append(f"POS Tags: {pos_tags}")
    return result

def lemma_example() -> str:
    """Demonstrate lemmatization."""
    nlp = initialize_pipeline("tokenize,mwt,pos,lemma")
    doc = nlp(SAMPLE_TEXT)
    result = ["Lemmatization Example:"]
    for sent in doc.sentences:
        lemmas = [(word.text, word.lemma) for word in sent.words]
        result.append(f"Sentence: {' '.join(word.text for word in sent.words)}")
        result.append(f"Lemmas: {lemmas}")
    return result

def depparse_example() -> str:
    """Demonstrate dependency parsing."""
    nlp = initialize_pipeline("tokenize,mwt,pos,lemma,depparse")
    doc = nlp(SAMPLE_TEXT)
    result = ["Dependency Parsing Example:"]
    for sent in doc.sentences:
        deps = [(word.text, word.deprel, word.head) for word in sent.words]
        result.append(f"Sentence: {' '.join(word.text for word in sent.words)}")
        result.append(f"Dependencies: {deps}")
    return result

def ner_example() -> str:
    """Demonstrate named entity recognition."""
    nlp = initialize_pipeline("tokenize,mwt,ner")
    doc = nlp(SAMPLE_TEXT)
    result = ["Named Entity Recognition Example:"]
    for sent in doc.sentences:
        entities = [(ent.text, ent.type) for ent in sent.ents]
        result.append(f"Sentence: {' '.join(word.text for word in sent.words)}")
        result.append(f"Entities: {entities}")
    return result

def sentiment_example() -> str:
    """Demonstrate sentiment analysis."""
    nlp = initialize_pipeline("tokenize,mwt,sentiment")
    doc = nlp(SAMPLE_TEXT)
    result = ["Sentiment Analysis Example:"]
    for i, sent in enumerate(doc.sentences):
        sentiment = sent.sentiment  # 0=negative, 1=neutral, 2=positive
        result.append(f"Sentence {i+1}: {' '.join(word.text for word in sent.words)}")
        result.append(f"Sentiment: {sentiment} ({['negative', 'neutral', 'positive'][sentiment]})")
    return result

def constituency_example() -> str:
    """Demonstrate constituency parsing."""
    nlp = initialize_pipeline("tokenize,mwt,pos,constituency")
    doc = nlp(SAMPLE_TEXT)
    result = ["Constituency Parsing Example:"]
    for sent in doc.sentences:
        result.append(f"Sentence: {' '.join(word.text for word in sent.words)}")
        result.append(f"Parse Tree: {sent.constituency}")
    return result

def save_results(results: List[str], output_path: str) -> None:
    """Save processor results to a file."""
    Path(output_path).write_text("\n\n".join(results), encoding="utf-8")

def main():
    """Run all processor examples and save results, each to a separate file."""
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # Download English and French models if not already present
    # stanza.download("en", model_dir=DEFAULT_MODEL_DIR)
    
    # Each example and its filename
    example_funcs = [
        (tokenize_example, "tokenize_example.json"),
        (mwt_example, "mwt_example.json"),
        (pos_example, "pos_example.json"),
        (lemma_example, "lemma_example.json"),
        (depparse_example, "depparse_example.json"),
        (ner_example, "ner_example.json"),
        (sentiment_example, "sentiment_example.json"),
        (constituency_example, "constituency_example.json"),
    ]
    
    saved_files = []
    for func, filename in example_funcs:
        results = func()
        output_path = os.path.join(output_dir, filename)
        save_results(results, output_path)
        saved_files.append(output_path)
        print(f"Result saved to {output_path}")
    
    # Optionally, summarize where the results were written
    print("\nAll example results saved in:")
    for file in saved_files:
        print(f"  {file}")

if __name__ == "__main__":
    main()