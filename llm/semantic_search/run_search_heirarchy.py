import shutil
from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL, OLLAMA_SMALL_LLM_MODEL
from jet.llm.query.retrievers import setup_index
from llama_index.core import Document, SimpleDirectoryReader, PromptTemplate
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

from convert_docs_to_scripts import scrape_code
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet._token import token_counter
from jet.llm.query import setup_deeplake_query, query_llm
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings, small_embed_model
initialize_ollama_settings(settings={"embedding_model": small_embed_model})


def get_token_counts(texts, model):
    token_counts = token_counter(
        texts, model, prevent_total=True)
    max_count = max(token_counts)
    min_count = min(token_counts)
    total_count = sum(token_counts)
    return {
        "max": max_count,
        "min": min_count,
        "total": total_count,
    }


if __name__ == "__main__":
    mode = "hierarchy"
    chunk_overlap = 40
    sub_chunk_sizes = [512, 128]
    path_or_docs = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"

    # Query config
    top_k = 20
    score_threshold = 0.0
    model = OLLAMA_SMALL_LLM_MODEL
    embed_model = OLLAMA_SMALL_EMBED_MODEL
    max_tokens = None

    sample_query = "Tell me about yourself."

    query_nodes = setup_index(
        path_or_docs, sub_chunk_sizes=sub_chunk_sizes, chunk_overlap=chunk_overlap, mode=mode)

    logger.newline()
    logger.info("HEIRARCHY SCORE: sample query...")
    result = query_nodes(
        sample_query, threshold=score_threshold, top_k=top_k)
    logger.info(f"RETRIEVED NODES ({len(result["nodes"])})")
    display_jet_source_nodes(sample_query, result["nodes"])

    result_texts = [node.text for node in result["nodes"]]
    token_count = get_token_counts(result_texts, model)

    logger.info("Token counts")
    logger.log("Max:", token_count['max'], colors=["DEBUG", "SUCCESS"])
    logger.log("Min:", token_count['min'], colors=["DEBUG", "SUCCESS"])
    logger.log("Total:", token_count['total'], colors=["DEBUG", "SUCCESS"])

    contexts = [
        f"File: {node.metadata['file_name']}\n{node.text}"
        for node in result["nodes"]
    ]

    response = query_llm(
        sample_query,
        contexts,
        model=model,
        max_tokens=max_tokens,
    )

    # Run app
    while True:
        # Continuously ask user for queries
        try:
            query = input("Enter your query (type 'exit' to quit): ").strip()
            if query.lower() == "exit":
                print("Exiting query loop.")
                break

            result = query_nodes(
                query, threshold=score_threshold, top_k=top_k)
            logger.info(f"RETRIEVED NODES ({len(result["nodes"])})")
            display_jet_source_nodes(query, result["nodes"])

            result_texts = [node.text for node in result["nodes"]]
            token_count = get_token_counts(result_texts, model)

            logger.info("Token counts")
            logger.log("Max:", token_count['max'], colors=["DEBUG", "SUCCESS"])
            logger.log("Min:", token_count['min'], colors=["DEBUG", "SUCCESS"])
            logger.log("Total:", token_count['total'], colors=[
                       "DEBUG", "SUCCESS"])

            # contexts = [f"File: {node.metadata['file']}\nPart: {
            #     node.metadata['part']}\n{node.text}" for node in result["nodes"]]
            contexts = [
                f"File: {node.metadata['file_name']}\n{node.text}"
                for node in result["nodes"]
            ]
            response = query_llm(
                query,
                contexts,
                model=model,
                max_tokens=max_tokens,
            )

        except KeyboardInterrupt:
            print("\nExiting query loop.")
            break
        except Exception as e:
            logger.error(f"Error while processing query: {e}")

    logger.info("\n\n[DONE]", bright=True)
