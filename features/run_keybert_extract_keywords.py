import os
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from jet.file.utils import load_file, save_file
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.vectors.document_types import HeaderDocument
from jet.wordnet.keywords.keyword_extraction import extract_query_candidates, extract_keywords_with_candidates, extract_keywords_with_custom_vectorizer, extract_keywords_with_embeddings, extract_multi_doc_keywords, extract_single_doc_keywords, setup_keybert, KeywordResult

if __name__ == "__main__":
    """Main function demonstrating KeyBERT usage."""
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    embed_model: EmbedModelType = "static-retrieval-mrl-en-v1"

    docs = load_file(docs_file)
    query = docs["query"]
    docs = HeaderDocument.from_list(docs["documents"])
    doc_texts = [text for doc in docs for text in doc.texts]

    # Sample documents
    single_doc = "\n".join(doc_texts)
    multi_docs = doc_texts

    candidate_keywords = extract_query_candidates(query)
    save_file(candidate_keywords, f"{output_dir}/candidate_keywords.json")

    # Initialize KeyBERT
    kw_model = setup_keybert()

    # Example 1: Extract keywords from a single document
    print("\nExample 1: Single Document Keywords")
    keywords = extract_single_doc_keywords(
        single_doc, kw_model, top_n=5, use_mmr=True, diversity=0.7, keyphrase_ngram_range=(1, 2))
    print(f"Keywords: {[(kw['text'], kw['score']) for kw in keywords]}")
    save_file(keywords, f"{output_dir}/extract_single_doc_keywords.json")

    # Example 2: Extract keywords from multiple documents
    print("\nExample 2: Multiple Documents Keywords")
    keywords = extract_multi_doc_keywords(
        multi_docs, kw_model, seed_keywords=candidate_keywords, top_n=5, use_mmr=True, diversity=0.7, keyphrase_ngram_range=(1, 2))
    for i, doc_keywords in enumerate(keywords):
        print(
            f"Document {i+1} Keywords: {[(kw['text'], kw['score']) for kw in doc_keywords]}")
    save_file(keywords, f"{output_dir}/extract_multi_doc_keywords.json")

    # Example 3: Extract keywords with candidate list
    print("\nExample 3: Keywords with Candidates")
    keywords = extract_keywords_with_candidates(
        multi_docs, kw_model, candidates=candidate_keywords, top_n=5)
    for i, doc_keywords in enumerate(keywords):
        print(
            f"Document {i+1} Keywords: {[(kw['text'], kw['score']) for kw in doc_keywords]}")
    save_file(keywords, f"{output_dir}/extract_keywords_with_candidates.json")

    # Example 4: Extract keywords with custom vectorizer
    print("\nExample 4: Keywords with Custom Vectorizer")
    custom_vectorizer = CountVectorizer(
        ngram_range=(1, 2), stop_words="english")
    keywords = extract_keywords_with_custom_vectorizer(
        multi_docs, kw_model, custom_vectorizer, seed_keywords=candidate_keywords, top_n=5)
    for i, doc_keywords in enumerate(keywords):
        print(
            f"Document {i+1} Keywords: {[(kw['text'], kw['score']) for kw in doc_keywords]}")
    save_file(
        keywords, f"{output_dir}/extract_keywords_with_custom_vectorizer.json")

    # Example 5: Extract keywords with precomputed embeddings
    print("\nExample 5: Keywords with Precomputed Embeddings")
    keywords = extract_keywords_with_embeddings(multi_docs, kw_model, top_n=5)
    for i, doc_keywords in enumerate(keywords):
        print(
            f"Document {i+1} Keywords: {[(kw['text'], kw['score']) for kw in doc_keywords]}")
    save_file(keywords, f"{output_dir}/extract_keywords_with_embeddings.json")
