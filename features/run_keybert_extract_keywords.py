import os
import shutil
from typing import List
from jet._token.token_utils import token_counter
from jet.models.utils import get_context_size
from sklearn.feature_extraction.text import CountVectorizer
from jet.file.utils import save_file
from jet.wordnet.keywords.helpers import extract_query_candidates, extract_keywords_with_candidates, extract_keywords_with_custom_vectorizer, extract_keywords_with_embeddings, extract_multi_doc_keywords, extract_single_doc_keywords, setup_keybert
from jet.libs.bertopic.examples.mock import load_sample_data

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


if __name__ == "__main__":
    """Main function demonstrating KeyBERT usage."""
    
    embed_model = "embeddinggemma"
    chunk_size = 96
    chunk_overlap = 32
    query = "Top isekai anime 2025"

    # Map HeaderDoc to texts and ids
    texts = load_sample_data(model=embed_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ids = [str(idx) for idx, doc in enumerate(texts)]
    save_file(texts, f"{OUTPUT_DIR}/documents.json")

    # Prepare single document for single_doc_keywords
    separator = "\n\n"
    sep_token_count = token_counter(separator, embed_model)
    token_counts: List[int] = token_counter(texts, embed_model, prevent_total=True)
    context_size = get_context_size(embed_model)
    save_file({
        "embed_model": embed_model,
        "query": query,
        "separator": separator,
        "docs_count": len(texts),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "context_size": context_size,
        "tokens": {
            "min": min(token_counts),
            "max": max(token_counts),
            "average": sum(token_counts) // len(token_counts),
            "total": sum(token_counts),
            "sep": sep_token_count
        }
    }, f"{OUTPUT_DIR}/_info.json")

    # Build single_doc up to <= context_size tokens, accounting for separator tokens
    tokens_so_far = 0
    single_doc_texts = []
    for idx, (text, num_tokens) in enumerate(zip(texts, token_counts)):
        tokens_to_add = num_tokens if isinstance(num_tokens, int) else sum(num_tokens)
        # Only add separator tokens if this is not the first line
        sep_count = sep_token_count if single_doc_texts else 0
        if tokens_so_far + sep_count + tokens_to_add > context_size:
            break
        if single_doc_texts:
            tokens_so_far += sep_count
        single_doc_texts.append(text)
        tokens_so_far += tokens_to_add
    single_doc = separator.join(single_doc_texts)
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
