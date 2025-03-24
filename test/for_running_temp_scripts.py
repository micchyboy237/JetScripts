from datetime import date
from urllib.parse import quote
from typing import List, Optional
from jet.logger import logger
from jet.search.formatters import clean_string
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.file.utils import load_file


def construct_google_query(
    title: str,
    properties: Optional[List[str]] = None,
    site: Optional[str] = None,
    exclude: Optional[List[str]] = None
) -> str:
    """
    Constructs a structured Google search query URL.

    :param title: The title of the anime.
    :param properties: A list of properties to search for (e.g., seasons, episodes, synopsis).
    :param site: A specific website to search in (e.g., "myanimelist.net").
    :param exclude: A list of words to exclude from search results.
    :return: A Google search URL string.
    """
    query = f'"{title}" anime'

    if properties:
        query += " " + " ".join(properties)

    if site:
        query += f" site:{site}"

    if exclude:
        query += " " + " ".join(f"-{word}" for word in exclude)

    return f"https://www.google.com/search?q={quote(query)}"


if __name__ == "__main__":
    from pydantic import BaseModel
    from jet.vectors.reranker.bm25_helpers import HybridSearch
    from jet.llm.query.retrievers import query_llm, setup_index
    from jet.llm.utils.llama_index_utils import display_jet_source_nodes
    from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
    from llama_index.core.schema import Document
    from jet.wordnet.similarity import filter_highest_similarity

    class Anime(BaseModel):
        title: str
        seasons: int
        episodes: int
        synopsis: Optional[str] = None
        genre: Optional[List[str]] = None
        release_date: Optional[date] = None
        end_date: Optional[date] = None

    # Get list of field names
    anime_fields = list(Anime.model_fields.keys())

    # model_name = "paraphrase-MiniLM-L12-v2"
    # model_name = "mxbai-embed-large"
    model_name = "nomic-embed-text"
    similarity_metric = "cosine"

    # data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    data = load_file(data_file)
    docs = []
    # for item in data:
    for item in [text for texts in list(data.values()) for text in texts]:
        cleaned_sentence = clean_string(item)
        docs.append(cleaned_sentence)

    search_keys_str = ", ".join(
        [key.replace('.', ' ').replace('_', ' ') for key in anime_fields])
    title = "I'll Become a Villainess Who Goes Down in History"
    query = f"Anime \"{title}\" {search_keys_str}"
    # query = "October seven is the date of our vacation to Camarines Sur."
    # docs = [
    #     "October 7 is our holiday in Camarines Sur.",
    #     "October 7 is the day we went on vacation to Camarines Sur.",
    #     "The seventh of October is the day of our vacation in Camarines Sur."
    # ]

    # result = filter_highest_similarity(
    #     query, docs, model_name=model_name, similarity_metric=similarity_metric)
    system = None
    documents = [Document(text=text) for text in docs]
    # Setup index
    query_nodes = setup_index(documents)
    # Search RAG nodes
    logger.newline()
    logger.info("RELATIVE_SCORE: sample query...")
    result = query_nodes(query, fusion_mode=FUSION_MODES.RELATIVE_SCORE)
    logger.info(f"RETRIEVED NODES ({len(result["nodes"])})")
    display_jet_source_nodes(query, result["nodes"])

    # Setup index
    hybrid_search = HybridSearch(model_name=model_name)
    hybrid_search.build_index(docs)

    # Search RAG nodes
    top_k = None
    threshold = 0.0
    search_results = hybrid_search.search(
        query, top_k=top_k, threshold=threshold)

    # Ask LLM
    llm_response_stream = query_llm(query, result['texts'], system=system)
    llm_response = ""
    for chunk in llm_response_stream:
        llm_response += chunk

    copy_to_clipboard(llm_response)

    logger.newline()
    logger.debug("LLM Response:")
    logger.success(llm_response)
