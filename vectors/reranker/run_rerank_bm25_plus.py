import os

from jet.data.sample_diverse_texts import sample_diverse_texts
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.vectors.document_types import HeaderDocument
from jet.vectors.reranker.bm25_plus import rerank_bm25_plus
from jet.wordnet.n_grams import count_ngrams


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    docs = [HeaderDocument(**doc) for doc in docs]

    query = "List top 10 isekai anime today."

    top_k: int = 10
    threshold: float = 0.8

    texts = [doc.text for doc in docs]

    word_counts = count_ngrams(
        "\n\n".join(texts), min_count=2, max_words=5, case_insensitive=True)
    save_file(word_counts, f"{output_dir}/word_counts.json")

    keywords = list(word_counts.items())
    save_file(keywords, f"{output_dir}/keywords.json")

    bm25_plus_result = rerank_bm25_plus(texts, query, keywords)
    save_file(bm25_plus_result, f"{output_dir}/bm25_plus_result.json")

    # Map doc_index to original texts and debug
    reranked_texts = []
    for result in bm25_plus_result["results"]:
        doc_index = result["doc_index"]
        score = result["score"]
        if score > threshold:
            original_text = texts[doc_index]
            reranked_texts.append(original_text)

    # Unique results and limit to top_k
    reranked_texts = list(dict.fromkeys(reranked_texts))
    # Get diverse texts
    logger.info(f"Sampling diverse texts ({len(reranked_texts)})...")
    reranked_texts: list[str] = sample_diverse_texts(reranked_texts)
    reranked_texts = reranked_texts[:top_k]
    save_file(reranked_texts, f"{output_dir}/reranked_texts.json")
