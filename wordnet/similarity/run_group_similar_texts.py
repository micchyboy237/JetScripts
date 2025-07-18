import os
import time

from fastapi.utils import generate_unique_id
from jet.file.utils import save_file, load_file
from jet.logger import logger
from jet.models.model_types import EmbedModelType
from jet.vectors.document_types import HeaderDocument
from jet.wordnet.similarity import group_similar_texts


if __name__ == '__main__':
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    docs = docs["documents"]
    documents = [
        f"{doc["header"].lstrip('#').strip()}\n{doc["content"]}" for doc in docs]

    model_name: EmbedModelType = "all-MiniLM-L12-v2"

    # Start timing
    start_time = time.time()

    ids = [doc["doc_id"] for doc in docs]

    grouped_similar_texts = group_similar_texts(
        documents, threshold=0.7, model_name=model_name, ids=ids)

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Log performance
    logger.log(f"group_similar_texts:",
               f"{execution_time:.2f}s", colors=["WHITE", "ORANGE"])

    save_file({"execution_time": f"{execution_time:.2f}s", "count": len(grouped_similar_texts), "results": grouped_similar_texts},
              f"{output_dir}/results.json")
