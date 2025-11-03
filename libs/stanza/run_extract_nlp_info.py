import os
import shutil
from typing import List
from jet.code.markdown_utils import base_parse_markdown, convert_markdown_to_text
import stanza
from tqdm import tqdm
from jet.libs.stanza.rag_stanza import StanzaPipelineCache
from jet.code.extraction.sentence_extraction import extract_sentences
from jet.adapters.stanza.ner_visualization import visualize_ner_str as visualize_ner
from jet.adapters.stanza.dependency_visualization import visualize_str as visualize_dep
from jet.adapters.stanza.semgrex_visualization import visualize_search_str as visualize_sem
from jet._token.token_utils import token_counter
from jet.file.utils import save_file
from jet.logger import logger
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

DEFAULT_MODEL_DIR = os.getenv(
    'STANZA_RESOURCES_DIR',
    os.path.join(os.path.expanduser("~/.cache"), "stanza_resources")
)

# Define the default set of processors for most complete English analyses
ALL_PROCESSORS = "tokenize,mwt,pos,lemma,depparse,ner,sentiment,constituency"

_nlp = None

def _load_pipeline(processors: str = ALL_PROCESSORS, lang: str = "en"):
    """Load Stanza NLP pipeline lazily, only once."""
    global _nlp
    if _nlp is None:
        logger.debug("Creating pipeline")
        _nlp = initialize_pipeline(processors, lang)
    else:
        logger.debug("Reusing cache for pipeline")
    return _nlp

def initialize_pipeline(processors: str = ALL_PROCESSORS, lang: str = "en") -> stanza.Pipeline:
    """Initialize Stanza pipeline with all processors."""
    cache = StanzaPipelineCache()
    # return stanza.Pipeline(lang=lang, dir=DEFAULT_MODEL_DIR, processors=processors, use_gpu=True)
    pipeline = cache.get_pipeline(
        lang=lang,
        processors=processors,
        use_gpu=True
    )
    return pipeline

def full_results_example(text: str) -> list:
    """Demonstrate tokenization and sentence segmentation."""
    logger.info("Tokenization Example:")
    nlp = _load_pipeline()
    doc = nlp(text)
    results = []
    for sent in doc.sentences:
        results.append(sent.to_dict())
    return results

def sentences_example(text: str) -> list:
    """Demonstrate tokenization and sentence segmentation."""
    logger.info("Tokenization Example:")
    nlp = _load_pipeline()
    doc = nlp(text)
    sentences = []
    for sent in doc.sentences:
        tokens = [token.text for token in sent.tokens]
        sentences.append(' '.join(tokens))
    return sentences

def tokenize_example(text: str) -> list:
    """Demonstrate tokenization and sentence segmentation."""
    logger.info("Tokenization Example:")
    nlp = _load_pipeline()
    doc = nlp(text)
    tokens_list = []
    for sent in doc.sentences:
        tokens = [token.text for token in sent.tokens]
        tokens_list.append(tokens)
    return tokens_list

def mwt_example(text: str) -> list:
    """Demonstrate multi-word token expansion for English."""
    logger.info("Multi-Word Token Expansion Example (English):")
    nlp = _load_pipeline()
    doc = nlp(text)  # English example with MWT (don't -> do + not)
    words_per_sentence = []
    for sent in doc.sentences:
        words = [word.text for word in sent.words]
        words_per_sentence.append(words)
    return words_per_sentence

def pos_example(text: str) -> list:
    """Demonstrate part-of-speech tagging."""
    logger.info("Part-of-Speech Tagging Example:")
    nlp = _load_pipeline()
    doc = nlp(text)
    sentences = []
    pos_tags_list = []
    for sent in doc.sentences:
        sentence_text = ' '.join(word.text for word in sent.words)
        pos_tags = [(word.text, word.upos, word.xpos, word.feats) for word in sent.words]
        sentences.append(sentence_text)
        pos_tags_list.append(pos_tags)
    return pos_tags_list

def lemma_example(text: str) -> list:
    """Demonstrate lemmatization."""
    logger.info("Lemmatization Example:")
    nlp = _load_pipeline()
    doc = nlp(text)
    sentences = []
    lemmas_list = []
    for sent in doc.sentences:
        sentence_text = ' '.join(word.text for word in sent.words)
        lemmas = [(word.text, word.lemma) for word in sent.words]
        sentences.append(sentence_text)
        lemmas_list.append(lemmas)
    return lemmas_list

def depparse_example(text: str) -> list:
    """Demonstrate dependency parsing."""
    logger.info("Dependency Parsing Example:")
    nlp = _load_pipeline()
    doc = nlp(text)
    sentences = []
    dependencies_list = []
    for sent in doc.sentences:
        sentence_text = ' '.join(word.text for word in sent.words)
        deps = [(word.text, word.deprel, word.head) for word in sent.words]
        sentences.append(sentence_text)
        dependencies_list.append(deps)
    return dependencies_list

def ner_example(text: str) -> list:
    """Demonstrate named entity recognition."""
    logger.info("Named Entity Recognition Example:")
    nlp = _load_pipeline()
    doc = nlp(text)
    sentences = []
    entities_list = []
    for sent in doc.sentences:
        sentence_text = ' '.join(word.text for word in sent.words)
        entities = [(ent.text, ent.type) for ent in sent.ents]
        sentences.append(sentence_text)
        entities_list.append(entities)
    return entities_list

def sentiment_example(text: str) -> list:
    """Demonstrate sentiment analysis."""
    logger.info("Sentiment Analysis Example:")
    nlp = _load_pipeline()
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
    return sentiment_list

def constituency_example(text: str) -> list:
    """Demonstrate constituency parsing."""
    logger.info("Constituency Parsing Example:")
    nlp = _load_pipeline()
    doc = nlp(text)
    sentences = []
    parse_trees = []
    for sent in doc.sentences:
        sentence_text = ' '.join(word.text for word in sent.words)
        sentences.append(sentence_text)
        parse_trees.append(str(sent.constituency))
    return parse_trees

def visualize_ner_example(text: str) -> str:
    nlp = _load_pipeline()
    return visualize_ner(text, nlp)

def visualize_dep_example(text: str) -> str:
    nlp = _load_pipeline()
    return visualize_dep(text, "en", nlp)

def visualize_sem_example(text: str) -> str:
    nlp = _load_pipeline()
    queries = ["{pos:NN}=object <obl {}=action",
               "{cpos:NOUN}=thing <obj {cpos:VERB}=action"]
    return visualize_sem(text, queries, "en", nlp)

text = """Title: Headhunted to Another World: From Salaryman to Big Four!
Isekai
Fantasy
Comedy
Release Date: January 1, 2025
Japanese Title: Salaryman ga Isekai ni Ittara Shitennou ni Natta Hanashi Studio

Studio: Geek Toys, CompTown

Based On: Manga

Creator: Benigashira

Streaming Service(s): Crunchyroll
Powered by
Expand Collapse
Plenty of 2025 isekai anime will feature OP protagonists capable of brute-forcing their way through any and every encounter, so it is always refreshing when an MC comes along that relies on brain rather than brawn. A competent office worker who feels underappreciated, Uchimura is suddenly summoned to another world by a demonic ruler, who comes with quite an unusual offer: Join the crew as one of the Heavenly Kings. So, Uchimura starts a new career path that tasks him with tackling challenges using his expertise in discourse and sales.
Related"""
texts = [text]

def main():
    """Run all processor examples on each document sequentially and save results with progress tracking."""
    # # Load all documents
    # chunks = load_sample_data_with_info(model="embeddinggemma", chunk_size=512, truncate=True)
    # save_file(chunks, f"{OUTPUT_DIR}/chunks.json")

    # token_counts = [chunk["num_tokens"] for chunk in chunks]
    # save_file({
    #     "chunks": len(chunks),
    #     "tokens": {
    #         "min": min(token_counts),
    #         "max": max(token_counts),
    #         "ave": sum(token_counts) // len(token_counts),
    #     }
    # }, f"{OUTPUT_DIR}/info.json")

    # headers = [{
    #     "doc_index": chunk["doc_index"],
    #     "content_tokens": chunk["num_tokens"],
    #     "header": chunk["meta"]["header"],
    # } for chunk in chunks]
    # save_file(headers, f"{OUTPUT_DIR}/headers.json")
    
    # Each example and its filename
    example_funcs = [
        (full_results_example, "full_results"),
        (sentences_example, "sentences"),
        (tokenize_example, "tokenize"),
        (mwt_example, "mwt"),
        (pos_example, "pos"),
        (lemma_example, "lemma"),
        (depparse_example, "depparse"),
        (ner_example, "ner"),
        (sentiment_example, "sentiment"),
        (constituency_example, "constituency"),
    ]

    example_visualization_funcs = [
        (visualize_ner_example, "visualize_ner"),
        (visualize_dep_example, "visualize_dep"),
        (visualize_sem_example, "visualize_sem"),
    ]
    
    # chunk_texts = [chunk["content"] for chunk in chunks]
    chunk_texts = texts
    saved_files = []
    # Process each document
    for doc_idx, md_content in enumerate(tqdm(chunk_texts, desc="Processing documents", unit="doc")):
        doc_dir = os.path.join(OUTPUT_DIR, f"doc_{doc_idx + 1}")
        os.makedirs(doc_dir, exist_ok=True)

        text = convert_markdown_to_text(md_content)
        sentences = extract_sentences(text, valid_only=True)

        if sentences:
            token_counts: List[int] = token_counter(sentences, "embeddinggemma", prevent_total=True)
            save_file(
                [{
                    "tokens": tokens,
                    "sentence": sentence
                } for tokens, sentence in zip(token_counts, sentences)],
                os.path.join(doc_dir, "sentences.json"),
            )

        save_file({
            "tokens": token_counter(text, "embeddinggemma"),
            "sentences": len(sentences),
        }, os.path.join(doc_dir, "info.json"))
        save_file(md_content, os.path.join(doc_dir, "doc.md"))
        save_file(base_parse_markdown(md_content, ignore_links=True), os.path.join(doc_dir, "md_tokens.json"))

        save_file(text, os.path.join(doc_dir, "doc.txt"))
        
        # Process all example functions for the current document
        for func, func_name in tqdm(example_funcs, desc=f"Processing tasks for doc_{doc_idx + 1}", unit="task", leave=False):
            results = func(text)
            output_path = os.path.join(doc_dir, "tasks", f"{func_name}.json")
            save_file([{"doc_id": doc_idx, "results": results}], output_path)
            saved_files.append(output_path)

        for func, func_name in tqdm(example_visualization_funcs, desc=f"Visualization for doc_{doc_idx + 1}", unit="visualization", leave=False):
            html = func(text)
            output_path = os.path.join(doc_dir, "visualization", f"{func_name}.html")
            save_file(html, output_path)
            saved_files.append(output_path)
    
    # Summarize where the results were written
    logger.gray("\nAll example results saved in:")
    for file in saved_files:
        logger.success(f"\n{file}", bright=True)

if __name__ == "__main__":
    main()