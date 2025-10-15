import os
import shutil
import stanza
from jet.file.utils import save_file
from jet.logger import logger

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

def tokenize_example() -> dict:
    """Demonstrate tokenization and sentence segmentation."""
    logger.info("Tokenization Example:")
    nlp = initialize_pipeline("tokenize")
    doc = nlp(SAMPLE_TEXT)
    sentences = []
    tokens_list = []
    for sent in doc.sentences:
        tokens = [token.text for token in sent.tokens]
        sentences.append(' '.join(tokens))
        tokens_list.append(tokens)
    return {
        "sentences": sentences,
        "tokens": tokens_list,
    }

def mwt_example() -> dict:
    """Demonstrate multi-word token expansion for English."""
    logger.info("Multi-Word Token Expansion Example (English):")
    nlp = initialize_pipeline("tokenize,mwt", lang="en")
    doc = nlp("I don't like to swim.")  # English example with MWT (don't -> do + not)
    tokens_per_sentence = []
    words_per_sentence = []
    for sent in doc.sentences:
        tokens = [token.text for token in sent.tokens]
        words = [word.text for word in sent.words]
        tokens_per_sentence.append(tokens)
        words_per_sentence.append(words)
    return {
        "tokens": tokens_per_sentence,
        "words": words_per_sentence,
    }

def pos_example() -> dict:
    """Demonstrate part-of-speech tagging."""
    logger.info("Part-of-Speech Tagging Example:")
    nlp = initialize_pipeline("tokenize,mwt,pos")
    doc = nlp(SAMPLE_TEXT)
    sentences = []
    pos_tags_list = []
    for sent in doc.sentences:
        sentence_text = ' '.join(word.text for word in sent.words)
        pos_tags = [(word.text, word.upos, word.xpos, word.feats) for word in sent.words]
        sentences.append(sentence_text)
        pos_tags_list.append(pos_tags)
    return {
        "sentences": sentences,
        "pos_tags": pos_tags_list,
    }

def lemma_example() -> dict:
    """Demonstrate lemmatization."""
    logger.info("Lemmatization Example:")
    nlp = initialize_pipeline("tokenize,mwt,pos,lemma")
    doc = nlp(SAMPLE_TEXT)
    sentences = []
    lemmas_list = []
    for sent in doc.sentences:
        sentence_text = ' '.join(word.text for word in sent.words)
        lemmas = [(word.text, word.lemma) for word in sent.words]
        sentences.append(sentence_text)
        lemmas_list.append(lemmas)
    return {
        "sentences": sentences,
        "lemmas": lemmas_list,
    }

def depparse_example() -> dict:
    """Demonstrate dependency parsing."""
    logger.info("Dependency Parsing Example:")
    nlp = initialize_pipeline("tokenize,mwt,pos,lemma,depparse")
    doc = nlp(SAMPLE_TEXT)
    sentences = []
    dependencies_list = []
    for sent in doc.sentences:
        sentence_text = ' '.join(word.text for word in sent.words)
        deps = [(word.text, word.deprel, word.head) for word in sent.words]
        sentences.append(sentence_text)
        dependencies_list.append(deps)
    return {
        "sentences": sentences,
        "dependencies": dependencies_list,
    }

def ner_example() -> dict:
    """Demonstrate named entity recognition."""
    logger.info("Named Entity Recognition Example:")
    nlp = initialize_pipeline("tokenize,mwt,ner")
    doc = nlp(SAMPLE_TEXT)
    sentences = []
    entities_list = []
    for sent in doc.sentences:
        sentence_text = ' '.join(word.text for word in sent.words)
        entities = [(ent.text, ent.type) for ent in sent.ents]
        sentences.append(sentence_text)
        entities_list.append(entities)
    return {
        "sentences": sentences,
        "entities": entities_list,
    }

def sentiment_example() -> dict:
    """Demonstrate sentiment analysis."""
    logger.info("Sentiment Analysis Example:")
    nlp = initialize_pipeline("tokenize,mwt,sentiment")
    doc = nlp(SAMPLE_TEXT)
    sentences = []
    sentiment_list = []
    for i, sent in enumerate(doc.sentences):
        sentence = ' '.join(word.text for word in sent.words)
        sentiment = sent.sentiment  # 0=negative, 1=neutral, 2=positive
        sentences.append(sentence)
        sentiment_list.append({
            "value": sentiment,
            "label": ["negative", "neutral", "positive"][sentiment]
        })
    return {
        "sentences": sentences,
        "sentiment": sentiment_list,
    }

def constituency_example() -> dict:
    """Demonstrate constituency parsing."""
    logger.info("Constituency Parsing Example:")
    nlp = initialize_pipeline("tokenize,mwt,pos,constituency")
    doc = nlp(SAMPLE_TEXT)
    sentences = []
    parse_trees = []
    for sent in doc.sentences:
        sentence_text = ' '.join(word.text for word in sent.words)
        sentences.append(sentence_text)
        parse_trees.append(str(sent.constituency))
    return {
        "sentences": sentences,
        "constituency_trees": parse_trees,
    }

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
        (tokenize_example, "tokenize_example"),
        (mwt_example, "mwt_example"),
        (pos_example, "pos_example"),
        (lemma_example, "lemma_example"),
        (depparse_example, "depparse_example"),
        (ner_example, "ner_example"),
        (sentiment_example, "sentiment_example"),
        (constituency_example, "constituency_example"),
    ]
    
    saved_files = []
    for func, func_name in example_funcs:
        results_dict = func()
        for key, results in results_dict.items():
            output_path = os.path.join(output_dir, func_name, f"{key}.json")
            save_file(results, output_path)
            saved_files.append(output_path)
    
    # Optionally, summarize where the results were written
    logger.gray("\nAll example results saved in:")
    for file in saved_files:
        logger.success(f"\n{file}", bright=True)

if __name__ == "__main__":
    main()