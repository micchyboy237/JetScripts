from typing import Optional
from jet.actions.vector_semantic_search import VectorSemanticSearch
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.logger import logger
from jet.scrapers.utils import clean_text
from jet.token.token_utils import filter_texts, get_model_max_tokens, split_texts, token_counter
from jet.transformers.formatters import format_json
from jet.file.utils import load_file, save_file
from jet.wordnet.similarity import search_similarities
from jet.llm.ollama.base import Ollama
from llama_index.core.prompts.base import PromptTemplate
from pydantic.main import BaseModel


if __name__ == "__main__":
    embed_model = "mxbai-embed-large"
    rerank_model = "all-minilm:33m"
    llm_model = "llama3.2"
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/langchain/cookbook/generated/RAPTOR/docs_texts.json"

    title = "I'll Become a Villainess Who Goes Down in History"
    query = f"How many seasons and episodes does \"{title}\" anime have?"

    # max_tokens = 0.5
    chunk_size = OLLAMA_MODEL_EMBEDDING_TOKENS[embed_model]
    chunk_overlap = 100
    chunk_buffer: int = token_counter(query, embed_model)
    top_k = 10

    data: list[str] = load_file(data_file)
    texts = [clean_text(text) for text in data]

    splitted_texts = split_texts(
        texts, embed_model, chunk_size, chunk_overlap, buffer=chunk_buffer)
    logger.log("splitted_texts:", len(
        splitted_texts), colors=["GRAY", "DEBUG"])

    # Vector search

    search = VectorSemanticSearch(
        candidates=splitted_texts, embed_model=embed_model)
    fusion_results = search.fusion_search(query)
    logger.newline()
    logger.orange(f"Fusion Search Results ({len(fusion_results)}):")

    embed_results: list[str] = []
    for query_idx, (query_line, group) in enumerate(fusion_results.items()):
        embed_results.extend([g["text"] for g in group])

        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])
    embed_results = embed_results[:top_k]
    # Rerank search
    chunk_size = get_model_max_tokens(rerank_model)
    chunk_overlap = 0

    rerank_candidates = split_texts(
        embed_results, rerank_model, chunk_size, chunk_overlap, buffer=chunk_buffer)
    reranked_results = search_similarities(
        query, candidates=rerank_candidates, model_name=rerank_model)
    logger.newline()
    logger.orange(f"Rerank Results ({len(reranked_results)}):")
    for result in reranked_results:
        logger.log("  +", f"{result['text'][:25]}:", f"{
            result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # LLM Query
    class AnimeDetails(BaseModel):
        seasons: int
        episodes: int
        additional_info: Optional[str] = None

    output_cls = AnimeDetails

    max_llm_tokens = 0.5
    contexts: list[str] = filter_texts(
        rerank_candidates, llm_model, max_llm_tokens)
    context = "\n\n".join(contexts)

    llm = Ollama(model=llm_model)
    qa_prompt = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information, schema and not prior knowledge, "
        "answer the query.\n"
        "The generated JSON must pass the provided schema when validated.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    response = llm.structured_predict(
        output_cls=output_cls,
        prompt=qa_prompt,
        context_str=context,
        query_str=query,
        llm_kwargs={
            "options": {"temperature": 0.3},
            # "max_prediction_ratio": 0.5
        },
    )
    logger.success(f"Seasons: {response.seasons}")
    logger.success(f"Episodes: {response.episodes}")
    logger.success(f"Additional Info: {response.additional_info}")
