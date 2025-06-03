import os
import shutil
from jet.data.sample_diverse_texts import sample_diverse_texts
from jet.features.nltk_utils import get_word_counts_lemmatized, get_word_sentence_combination_counts
from jet.file.utils import load_file, save_file
from jet.llm.utils.bm25_plus import bm25_plus
from jet.logger import logger

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"

    # Load JSON data
    docs = load_file(docs_file)
    print(f"Loaded JSON data {len(docs)} from: {docs_file}")
    docs = [doc["text"] for doc in docs]
    docs_str = "\n\n".join(docs)

    # Process word counts for each document as a whole
    word_counts_lemmatized_text_results = get_word_counts_lemmatized(
        docs_str, min_count=10, as_score=False)
    output_path = f"{output_dir}/word_counts_lemmatized_counts.json"
    save_file(word_counts_lemmatized_text_results, output_path)

    # Get all keywords from word counts
    keywords = list(word_counts_lemmatized_text_results.keys())

    # Filter keywords by length (>= 3 chars) and take top 20
    keywords = [keyword for keyword in keywords if len(keyword) >= 3]
    keywords = keywords[:20]

    # Additional length filter (> 2 chars) and remove duplicates
    keywords = [keyword for keyword in keywords if len(keyword) > 2]
    keywords = list(set(keywords))

    # Log and save keywords
    logger.info(f"Keywords: {keywords}")
    output_path = f"{output_dir}/keywords.json"
    save_file(keywords, output_path)

    query = " ".join(keywords)
    logger.info(f"Query: {query}")
    logger.info(f"Reranking docs ({len(docs)})...")
    bm25_plus_results = bm25_plus(docs, query, k1=1.5)
    save_file(bm25_plus_results, f"{output_dir}/bm25_plus_results.json")

    # Map doc_index to original docs
    reranked_docs = []
    for result in bm25_plus_results["results"]:
        doc_index = result["doc_index"]
        score = result["score"]
        if score >= 0.9:
            reranked_docs.append(docs[doc_index])

    # Save reranked docs
    output_path = f"{output_dir}/reranked_docs.json"
    save_file(reranked_docs, output_path)

    # Get diverse docs
    reranked_diverse_docs = sample_diverse_texts(reranked_docs)
    output_path = f"{output_dir}/reranked_diverse_docs.json"
    save_file(reranked_diverse_docs, output_path)
