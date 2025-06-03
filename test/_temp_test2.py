import os
import shutil
from typing import List
from jet.code.splitter_markdown_utils import get_header_level
from jet.data.sample_diverse_texts import sample_diverse_texts
from jet.features.nltk_search import PosTag, get_pos_tag
from jet.features.nlp_utils import get_word_counts_lemmatized, get_word_sentence_combination_counts
from jet.file.utils import load_file, save_file
from jet.llm.utils.bm25_plus import bm25_plus, bm25_plus_with_keyword_counts
from jet.llm.utils.search_docs import search_docs
from jet.logger import logger
from jet.wordnet.n_grams import count_ngrams
from jet.wordnet.similarity import query_similarity_scores

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    query = "List all ongoing and upcoming isekai anime 2025."
    query_pos_tags: List[PosTag] = get_pos_tag(query)
    pos_mapping = {
        'N': 'noun',
        'V': 'verb',
        'J': 'adjective',
        'R': 'adverb',
        'C': 'number'
    }
    query_pos = list(set([pos_mapping[pos["pos"][0].upper()]
                     for pos in query_pos_tags
                     if pos["pos"][0].upper() in pos_mapping]))
    query_words = [tag["word"]for tag in query_pos_tags]

    output_path = f"{output_dir}/query_info.json"
    save_file({
        "query": query,
        "pos": query_pos,
        "pos_tags": query_pos_tags,
    }, output_path)

    # Load JSON data
    docs = load_file(docs_file)
    docs = [doc for doc in docs if doc["metadata"]["header_level"] in [2, 3]]
    texts = [doc["text"] for doc in docs]

    print(f"Loaded JSON data {len(docs)} from: {docs_file}")
    formatted_docs = [
        # f"{doc["text"]}" for doc in docs
        f"{doc["metadata"]["parent_header"] or ""}\n{doc["text"]}".strip() for doc in docs
    ]
    docs_str = "\n\n".join(formatted_docs)

    # Process word counts for each document as a whole
    # word_counts = get_word_counts_lemmatized(
    #     docs_str, pos=query_pos, min_count=10, as_score=False)
    # output_path = f"{output_dir}/word_counts.json"
    # save_file(word_counts, output_path)
    word_counts = count_ngrams(
        docs_str, min_count=10, max_words=1, case_insensitive=True)
    output_path = f"{output_dir}/word_counts.json"
    save_file(word_counts, output_path)

    # Get all keywords from word counts
    keywords = list(word_counts.items())
    all_doc_words = [word for word, count in keywords]

    vector_search_results = search_docs(
        query, formatted_docs, model="mxbai-embed-large")
    save_file({"query": query, "results": vector_search_results},
              f"{output_dir}/vector_search_results.json")

    # Filter keywords by length (>= 3 chars)
    keywords = {
        keyword: count
        for keyword, count in keywords
        if len(keyword) >= 3
    }

    # Log and save keywords
    logger.info(f"Keywords: {len(keywords)}")
    output_path = f"{output_dir}/keywords.json"
    save_file(keywords, output_path)

    logger.info(f"Reranking docs ({len(docs)})...")
    bm25_plus_results = bm25_plus(texts, query, k1=1.5)
    # bm25_plus_results = bm25_plus_with_keyword_counts(
    #     texts, keywords, query=query, k1=1.5)
    save_file(bm25_plus_results, f"{output_dir}/bm25_plus_results.json")

    # Map doc_index to original docs
    reranked_docs = []
    for result in bm25_plus_results["results"]:
        doc_index = result["doc_index"]
        score = result["score"]
        if score >= 0.7:
            reranked_docs.append(result)

    # Save reranked docs
    output_path = f"{output_dir}/reranked_docs.json"
    save_file(reranked_docs, output_path)

    # Get diverse docs
    reranked_diverse_docs = sample_diverse_texts(
        [doc["text"] for doc in reranked_docs])
    output_path = f"{output_dir}/reranked_diverse_docs.json"
    save_file(reranked_diverse_docs, output_path)
