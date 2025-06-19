import os
import time
from typing import List, Tuple, Dict, TypedDict
from jet.file.utils import save_file, load_file
from jet.logger import logger
from jet.models.model_types import EmbedModelType
from jet.vectors.document_types import HeaderDocument
from jet.wordnet.similarity import get_similar_texts

# Define typed structures for clarity


if __name__ == '__main__':
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load documents
    try:
        docs = load_file(docs_file)
        query = docs["query"]
        docs = docs["documents"]
        docs = [HeaderDocument(**doc) for doc in docs]
        logger.log("main:", f"Loaded {len(docs)} documents", colors=[
                   "WHITE", "BLUE"])
    except Exception as e:
        logger.log("main:", f"Failed to load documents: {str(e)}", colors=[
                   "WHITE", "RED"])
        raise

    model_name: EmbedModelType = "all-MiniLM-L12-v2"

    # Start timing
    start_time = time.time()

    # Group similar documents
    try:
        texts = [doc["header"].lstrip('#').strip() for doc in docs]
        grouped_results = get_similar_texts(texts, threshold=0.7)
    except Exception as e:
        logger.log("main:", f"Grouping failed: {str(e)}", colors=[
                   "WHITE", "RED"])
        raise

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Log performance
    logger.log(
        "main:",
        f"get_similar_texts: {execution_time:.2f}s, groups: {len(grouped_results)}",
        colors=["WHITE", "ORANGE"]
    )

    # Save results
    output_data = {
        "execution_time": f"{execution_time:.2f}s",
        "count": len(grouped_results),
        "results": grouped_results
    }
    try:
        save_file(output_data, f"{output_dir}/results.json")
        logger.log("main:", "Results saved successfully",
                   colors=["WHITE", "GREEN"])
    except Exception as e:
        logger.log("main:", f"Failed to save results: {str(e)}", colors=[
                   "WHITE", "RED"])
        raise
