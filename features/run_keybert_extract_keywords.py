import os
import shutil
from typing import List
from jet.code.markdown_types import HeaderDoc
from jet.logger import logger
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.utils.url_utils import clean_links
from jet.wordnet.text_chunker import chunk_texts
from sklearn.feature_extraction.text import CountVectorizer
from jet.file.utils import load_file, save_file
from jet.wordnet.keywords.helpers import extract_query_candidates, extract_keywords_with_candidates, extract_keywords_with_custom_vectorizer, extract_keywords_with_embeddings, extract_multi_doc_keywords, extract_single_doc_keywords, setup_keybert


OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def load_sample_data():
    """Load sample dataset from local for topic modeling."""
    embed_model = "embeddinggemma"
    headers_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/all_headers.json"
    
    logger.info("Loading sample dataset...")
    headers_dict = load_file(headers_file)
    # headers: List[HeaderDoc] = [h for h_list in headers_dict.values() for h in h_list]
    headers: List[HeaderDoc] = headers_dict["https://gamerant.com/new-isekai-anime-2025"]
    documents = [f"{doc["header"]}\n\n{doc['content']}" for doc in headers]

    # Clean all links
    documents = [clean_markdown_links(doc) for doc in documents]
    documents = [clean_links(doc) for doc in documents]

    documents = chunk_texts(
        documents,
        chunk_size=64,
        chunk_overlap=32,
        model=embed_model,
    )
    save_file(documents, f"{OUTPUT_DIR}/documents.json")
    return documents

if __name__ == "__main__":
    """Main function demonstrating KeyBERT usage."""
    
    embed_model = "all-MiniLM-L6-v2"
    query = "Top isekai anime 2025"

    # Map HeaderDoc to texts and ids
    texts = load_sample_data()
    ids = [str(idx) for idx, doc in enumerate(texts)]
    save_file(texts, f"{OUTPUT_DIR}/documents.json")

    # Prepare single document for single_doc_keywords
    single_doc = "\n".join(texts)
    single_doc_id = "combined_doc"

    # Extract candidate keywords
    candidate_keywords = extract_query_candidates(query)
    save_file(candidate_keywords, f"{OUTPUT_DIR}/candidate_keywords.json")

    # Setup KeyBERT model
    kw_model = setup_keybert(embed_model)

    print("\nExample 1: Single Document Keywords")
    keywords = extract_single_doc_keywords(
        single_doc, kw_model, id=single_doc_id, seed_keywords=candidate_keywords, top_n=5, use_mmr=True, diversity=0.7, keyphrase_ngram_range=(1, 3))
    for result in keywords:
        print(
            f"Doc ID: {result['id']}, Keywords: {[(kw['text'], kw['score']) for kw in result['keywords']]}, Tokens: {result['tokens']}")
    save_file(keywords, f"{OUTPUT_DIR}/extract_single_doc_keywords.json")

    print("\nExample 2: Multiple Documents Keywords")
    keywords = extract_multi_doc_keywords(
        texts, kw_model, ids=ids, seed_keywords=candidate_keywords, top_n=5, use_mmr=True, diversity=0.7, keyphrase_ngram_range=(1, 3))
    for result in keywords:
        print(
            f"Doc ID: {result['id']}, Keywords: {[(kw['text'], kw['score']) for kw in result['keywords']]}, Tokens: {result['tokens']}")
    save_file(keywords, f"{OUTPUT_DIR}/extract_multi_doc_keywords.json")

    print("\nExample 3: Keywords with Candidates")
    keywords = extract_keywords_with_candidates(
        texts, kw_model, candidates=candidate_keywords, ids=ids, seed_keywords=candidate_keywords, top_n=5, keyphrase_ngram_range=(1, 3))
    for result in keywords:
        print(
            f"Doc ID: {result['id']}, Keywords: {[(kw['text'], kw['score']) for kw in result['keywords']]}, Tokens: {result['tokens']}")
    save_file(keywords, f"{OUTPUT_DIR}/extract_keywords_with_candidates.json")

    print("\nExample 4: Keywords with Custom Vectorizer")
    custom_vectorizer = CountVectorizer(
        ngram_range=(1, 3), stop_words="english")
    keywords = extract_keywords_with_custom_vectorizer(
        texts, kw_model, custom_vectorizer, ids=ids, seed_keywords=candidate_keywords, top_n=5)
    for result in keywords:
        print(
            f"Doc ID: {result['id']}, Keywords: {[(kw['text'], kw['score']) for kw in result['keywords']]}, Tokens: {result['tokens']}")
    save_file(
        keywords, f"{OUTPUT_DIR}/extract_keywords_with_custom_vectorizer.json")

    print("\nExample 5: Keywords with Precomputed Embeddings")
    keywords = extract_keywords_with_embeddings(
        texts, kw_model, ids=ids, seed_keywords=candidate_keywords, top_n=5, keyphrase_ngram_range=(1, 3))
    for result in keywords:
        print(
            f"Doc ID: {result['id']}, Keywords: {[(kw['text'], kw['score']) for kw in result['keywords']]}, Tokens: {result['tokens']}")
    save_file(keywords, f"{OUTPUT_DIR}/extract_keywords_with_embeddings.json")
