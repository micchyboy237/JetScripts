
from jet.cache.joblib.utils import load_or_save_cache, save_cache
from jet.file.utils import load_file
from jet.logger import logger
from jet.vectors.reranker.bm25_helpers import HybridSearch, preprocess_texts, split_text_by_docs, transform_queries_to_ngrams
from jet.wordnet.n_grams import extract_ngrams, count_ngrams


def build_ngrams(texts: list[str], max_tokens: int):
    preprocessed_texts = preprocess_texts(texts)

    data = texts
    docs = split_text_by_docs(preprocessed_texts, max_tokens)
    ids = [str(idx) for idx, doc in enumerate(docs)]
    doc_texts = [doc.text for doc in docs]

    ngrams = count_ngrams(
        [text.lower() for text in doc_texts], min_words=1)

    return ngrams


if __name__ == "__main__":
    results = preprocess_texts(
        "Seasons, episodes and synopsis of \"I'll Become a Villainess Who Goes Down in History\" anime")
    logger.success(results)

if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    texts: list[str] = load_file(data_file)
    max_tokens = 256

    ngrams_cache_path = "generated/ngrams"
    ngrams: dict[str, int] = load_or_save_cache(ngrams_cache_path)
    if not ngrams:
        ngrams = build_ngrams(texts, max_tokens)
        save_cache(ngrams_cache_path, ngrams)

    query = "Seasons, episodes and synopsis of \"I'll Become a Villainess Who Goes Down in History\" anime"
    queries = transform_queries_to_ngrams(query.lower(), ngrams)
    assert len(queries) > 1


if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    texts: list[str] = load_file(data_file)

    hybrid_search = HybridSearch()
    hybrid_search.build_index(texts)

    # queries = ["Season", "episode", "synopsis"]
    query = "Seasons, episodes and synopsis of \"I'll Become a Villainess Who Goes Down in History\" anime"
    top_k = 10
    results = hybrid_search.search(query, top_k=top_k)
