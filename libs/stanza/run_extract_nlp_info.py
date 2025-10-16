import os
import shutil
import stanza
from tqdm import tqdm
from jet.libs.bertopic.examples.mock import load_sample_data
from jet.file.utils import save_file
from jet.logger import logger

DEFAULT_MODEL_DIR = os.getenv(
    'STANZA_RESOURCES_DIR',
    os.path.join(os.path.expanduser("~/.cache"), "stanza_resources")
)

def initialize_pipeline(processors: str, lang: str = "en") -> stanza.Pipeline:
    """Initialize Stanza pipeline with specified processors."""
    return stanza.Pipeline(lang=lang, dir=DEFAULT_MODEL_DIR, processors=processors, use_gpu=True)

def tokenize_example(text: str) -> dict:
    """Demonstrate tokenization and sentence segmentation."""
    logger.info("Tokenization Example:")
    nlp = initialize_pipeline("tokenize")
    doc = nlp(text)
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

def mwt_example(text: str) -> dict:
    """Demonstrate multi-word token expansion for English."""
    logger.info("Multi-Word Token Expansion Example (English):")
    nlp = initialize_pipeline("tokenize,mwt", lang="en")
    doc = nlp(text)  # English example with MWT (don't -> do + not)
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

def pos_example(text: str) -> dict:
    """Demonstrate part-of-speech tagging."""
    logger.info("Part-of-Speech Tagging Example:")
    nlp = initialize_pipeline("tokenize,mwt,pos")
    doc = nlp(text)
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

def lemma_example(text: str) -> dict:
    """Demonstrate lemmatization."""
    logger.info("Lemmatization Example:")
    nlp = initialize_pipeline("tokenize,mwt,pos,lemma")
    doc = nlp(text)
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

def depparse_example(text: str) -> dict:
    """Demonstrate dependency parsing."""
    logger.info("Dependency Parsing Example:")
    nlp = initialize_pipeline("tokenize,mwt,pos,lemma,depparse")
    doc = nlp(text)
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

def ner_example(text: str) -> dict:
    """Demonstrate named entity recognition."""
    logger.info("Named Entity Recognition Example:")
    nlp = initialize_pipeline("tokenize,mwt,ner")
    doc = nlp(text)
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

def sentiment_example(text: str) -> dict:
    """Demonstrate sentiment analysis."""
    logger.info("Sentiment Analysis Example:")
    nlp = initialize_pipeline("tokenize,mwt,sentiment")
    doc = nlp(text)
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

def constituency_example(text: str) -> dict:
    """Demonstrate constituency parsing."""
    logger.info("Constituency Parsing Example:")
    nlp = initialize_pipeline("tokenize,mwt,pos,constituency")
    doc = nlp(text)
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
    """Run all processor examples on each document sequentially and save results with progress tracking."""
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load all documents
    docs = load_sample_data(model="embeddinggemma", chunk_size=128, truncate=True)
    save_file(docs, f"{output_dir}/docs.json")
    
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
    # Process each document
    for doc_idx, text in enumerate(tqdm(docs, desc="Processing documents", unit="doc")):
        doc_dir = os.path.join(output_dir, f"doc_{doc_idx + 1}")
        os.makedirs(doc_dir, exist_ok=True)
        
        # Process all example functions for the current document
        for func, func_name in tqdm(example_funcs, desc=f"Processing tasks for doc_{doc_idx + 1}", unit="task", leave=False):
            results_dict = func(text)
            for key, results in results_dict.items():
                output_path = os.path.join(doc_dir, f"{func_name}_{key}.json")
                save_file([{"doc_id": doc_idx, "results": results}], output_path)
                saved_files.append(output_path)
    
    # Summarize where the results were written
    logger.gray("\nAll example results saved in:")
    for file in saved_files:
        logger.success(f"\n{file}", bright=True)

if __name__ == "__main__":
    main()