import os
import time
from jet.file.utils import save_file, load_file
from jet.logger import logger
from jet.models.model_types import EmbedModelType
from jet.wordnet.similarity import group_similar_texts


if __name__ == '__main__':
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    documents = [
        "\n".join([
            # doc["metadata"].get("parent_header") or "",
            doc["metadata"]["header"].lstrip('#').strip(),
            # doc["metadata"]["content"]
        ]).strip()
        for doc in docs
    ]
    model_name: EmbedModelType = "all-MiniLM-L12-v2"

    # Start timing
    start_time = time.time()

    grouped_similar_texts = group_similar_texts(
        documents, threshold=0.5, model_name=model_name)

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Log performance
    logger.log(f"group_similar_texts:",
               f"{execution_time:.2f}s", colors=["WHITE", "ORANGE"])

    save_file({"execution_time": f"{execution_time:.2f}s", "count": len(grouped_similar_texts), "results": grouped_similar_texts},
              f"{output_dir}/grouped_similar_texts.json")
