import os
from typing import List, Optional, TypedDict
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from jet.code.markdown_utils import parse_markdown
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.vectors.document_types import HeaderDocument
from jet.wordnet.keywords.keyword_extraction import extract_query_candidates, extract_keywords_with_candidates, extract_keywords_with_custom_vectorizer, extract_keywords_with_embeddings, extract_multi_doc_keywords, extract_single_doc_keywords, setup_keybert, SimilarityResult


class HeaderDoc(TypedDict):
    doc_id: str
    header: str
    content: str
    level: Optional[int]
    parent_header: Optional[str]
    parent_level: Optional[int]


if __name__ == "__main__":
    """Main function demonstrating KeyBERT usage."""
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/pages/gamerant_com_new_isekai_anime_2025/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    embed_model: EmbedModelType = "static-retrieval-mrl-en-v1"

    # Load HeaderDoc objects
    docs: List[HeaderDoc] = load_file(docs_file)
    query = "Top isekai anime 2025."

    # Map HeaderDoc to texts and ids
    texts = [f"{doc['header']}\n{doc['content']}" for doc in docs]
    ids = [doc['doc_id'] for doc in docs]

    # Prepare single document for single_doc_keywords
    single_doc = "\n".join(texts)
    single_doc_id = "combined_doc"

    # Extract candidate keywords
    candidate_keywords = extract_query_candidates(query)
    save_file(candidate_keywords, f"{output_dir}/candidate_keywords.json")

    # Setup KeyBERT model
    kw_model = setup_keybert()

    print("\nExample 1: Single Document Keywords")
    keywords = extract_single_doc_keywords(
        single_doc, kw_model, id=single_doc_id, seed_keywords=candidate_keywords, top_n=5, use_mmr=True, diversity=0.7, keyphrase_ngram_range=(1, 3))
    for result in keywords:
        print(
            f"Doc ID: {result['id']}, Keywords: {[(kw['text'], kw['score']) for kw in result['keywords']]}, Tokens: {result['tokens']}")
    save_file(keywords, f"{output_dir}/extract_single_doc_keywords.json")

    print("\nExample 2: Multiple Documents Keywords")
    keywords = extract_multi_doc_keywords(
        texts, kw_model, ids=ids, seed_keywords=candidate_keywords, top_n=5, use_mmr=True, diversity=0.7, keyphrase_ngram_range=(1, 3))
    for result in keywords:
        print(
            f"Doc ID: {result['id']}, Keywords: {[(kw['text'], kw['score']) for kw in result['keywords']]}, Tokens: {result['tokens']}")
    save_file(keywords, f"{output_dir}/extract_multi_doc_keywords.json")

    print("\nExample 3: Keywords with Candidates")
    keywords = extract_keywords_with_candidates(
        texts, kw_model, candidates=candidate_keywords, ids=ids, seed_keywords=candidate_keywords, top_n=5, keyphrase_ngram_range=(1, 3))
    for result in keywords:
        print(
            f"Doc ID: {result['id']}, Keywords: {[(kw['text'], kw['score']) for kw in result['keywords']]}, Tokens: {result['tokens']}")
    save_file(keywords, f"{output_dir}/extract_keywords_with_candidates.json")

    print("\nExample 4: Keywords with Custom Vectorizer")
    custom_vectorizer = CountVectorizer(
        ngram_range=(1, 3), stop_words="english")
    keywords = extract_keywords_with_custom_vectorizer(
        texts, kw_model, custom_vectorizer, ids=ids, seed_keywords=candidate_keywords, top_n=5)
    for result in keywords:
        print(
            f"Doc ID: {result['id']}, Keywords: {[(kw['text'], kw['score']) for kw in result['keywords']]}, Tokens: {result['tokens']}")
    save_file(
        keywords, f"{output_dir}/extract_keywords_with_custom_vectorizer.json")

    print("\nExample 5: Keywords with Precomputed Embeddings")
    keywords = extract_keywords_with_embeddings(
        texts, kw_model, ids=ids, seed_keywords=candidate_keywords, top_n=5, keyphrase_ngram_range=(1, 3))
    for result in keywords:
        print(
            f"Doc ID: {result['id']}, Keywords: {[(kw['text'], kw['score']) for kw in result['keywords']]}, Tokens: {result['tokens']}")
    save_file(keywords, f"{output_dir}/extract_keywords_with_embeddings.json")
