import os
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from sklearn.feature_extraction.text import CountVectorizer
from jet.file.utils import load_file, save_file
from jet.wordnet.keywords.helpers import extract_query_candidates, extract_keywords_with_candidates, extract_keywords_with_custom_vectorizer, extract_keywords_with_embeddings, extract_multi_doc_keywords, extract_single_doc_keywords, setup_keybert


if __name__ == "__main__":
    """Main function demonstrating KeyBERT usage."""
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/https_docs_tavily_com_documentation_api_reference_endpoint_crawl/page.html"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    
    embed_model = "all-MiniLM-L6-v2"
    query: str = "How to change max depth?"

    html_str = load_file(html_file)
    save_file(html_str, f"{output_dir}/formats/page.html")

    doc_markdown = convert_html_to_markdown(html_str, ignore_links=False)
    save_file(doc_markdown, f"{output_dir}/formats/markdown.md")

    documents = derive_by_header_hierarchy(doc_markdown, ignore_links=True)
    save_file(documents, f"{output_dir}/documents.json")
    
    # Map HeaderDoc to texts and ids
    texts = [f"{doc['header']}\n{doc['content']}" for doc in documents]
    ids = [doc['id'] for doc in documents]

    # Prepare single document for single_doc_keywords
    single_doc = "\n".join(texts)
    single_doc_id = "combined_doc"

    # Extract candidate keywords
    candidate_keywords = extract_query_candidates(query)
    save_file(candidate_keywords, f"{output_dir}/candidate_keywords.json")

    # Setup KeyBERT model
    kw_model = setup_keybert(embed_model)

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
