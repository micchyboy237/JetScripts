from datetime import date
from jet.llm.query.retrievers import query_llm, query_llm_structured
from jet.transformers.formatters import format_json
from pydantic import BaseModel
from typing import List, Optional
from jet.cache.joblib.utils import load_or_save_cache, save_cache
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.utils.commands import copy_to_clipboard
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


# if __name__ == "__main__":
#     data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
#     texts: list[str] = load_file(data_file)
#     max_tokens = 256

#     ngrams_cache_path = "generated/ngrams"
#     ngrams: dict[str, int] = load_or_save_cache(ngrams_cache_path)
#     if not ngrams:
#         ngrams = build_ngrams(texts, max_tokens)
#         save_cache(ngrams_cache_path, ngrams)

#     query = "Seasons, episodes and synopsis of \"I'll Become a Villainess Who Goes Down in History\" anime"
#     queries = transform_queries_to_ngrams(query.lower(), ngrams)
#     assert len(queries) > 1


if __name__ == "__main__":
    embed_model = "nomic-embed-text"
    llm_model = "llama3.1"
    system = None

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    data: dict[str, list[str]] = load_file(data_file)
    texts = [text for texts in list(data.values()) for text in texts]

    hybrid_search = HybridSearch(model_name=embed_model)
    hybrid_search.build_index(texts)

    # class Anime(BaseModel):
    #     title: str
    #     seasons: int
    #     episodes: int
    #     synopsis: Optional[str] = None
    #     genre: Optional[List[str]] = None
    #     release_date: Optional[date] = None
    #     end_date: Optional[date] = None

    class Episode(BaseModel):
        season: int
        episode: int
        date: date
        title: Optional[str] = None
        summary: Optional[str] = None

    class AnimeDetails(BaseModel):
        title: str
        seasons: Optional[int] = None
        status: Optional[str] = None
        synopsis: Optional[str] = None
        genre: Optional[List[str]] = None
        release_date: Optional[date] = None
        end_date: Optional[date] = None
        episodes: Optional[List[Episode]] = None

    output_cls = AnimeDetails
    # Get list of field names
    anime_fields = list(output_cls.model_fields.keys())
    search_keys_str = ", ".join(
        [key.replace('.', ' ').replace('_', ' ') for key in anime_fields])
    title = "I'll Become a Villainess Who Goes Down in History"
    query = f"Anime \"{title}\" {search_keys_str}"
    # query = "Episode 11 of \"I'll Become a Villainess Who Goes Down in History\" anime"
    # query = "\"I'll Become a Villainess Who Goes Down in History\" genre"

    top_k = None
    threshold = 0.0
    results = hybrid_search.search(query, top_k=top_k, threshold=threshold)
    results = results.copy()

    semantic_results = results.pop("semantic_results")
    reranked_results = results.pop("results")

    save_file(results, "generated/hybrid_search/results_info.json")
    save_file(semantic_results, "generated/hybrid_search/semantic_results.json")
    save_file(reranked_results, "generated/hybrid_search/reranked_results.json")

    # Ask LLM
    texts = [result["text"] for result in reranked_results]
    llm_response_stream = query_llm_structured(
        query, texts, model=llm_model, system=system, output_cls=output_cls)
    structued_llm_results = []
    for structued_llm_response in llm_response_stream:
        structued_llm_results.append(structued_llm_response)

        save_file(structued_llm_results,
                  "generated/hybrid_search/structued_llm_results.json")

    logger.newline()
    logger.success(f"Structured LLM Results: {len(structued_llm_results)}")
