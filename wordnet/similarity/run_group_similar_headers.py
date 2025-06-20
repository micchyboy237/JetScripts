import os
import time
from jet.data.sample_diverse_texts import sample_diverse_texts
from jet.file.utils import save_file, load_file
from jet.logger import logger
from jet.models.model_types import EmbedModelType
from jet.vectors.document_types import HeaderDocumentWithScore
from jet.wordnet.similarity import group_similar_headers


if __name__ == '__main__':
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/search_doc_results.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load documents
    try:
        docs = load_file(docs_file)
        query = docs["query"]
        docs = docs["results"]
        docs = [HeaderDocumentWithScore(**doc) for doc in docs]
        logger.log("main:", f"Loaded {len(docs)} documents", colors=[
                   "WHITE", "BLUE"])
    except Exception as e:
        logger.log("main:", f"Failed to load documents: {str(e)}", colors=[
                   "WHITE", "RED"])
        raise

    model_name: EmbedModelType = "static-retrieval-mrl-en-v1"

    # Start timing
    start_time = time.time()

    # Group similar documents
    grouped_results = group_similar_headers(
        docs, threshold=0.7, model_name=model_name)

    # Sort each group's "documents" by score in descending order
    for group in grouped_results:
        group_docs = group.get("documents", [])
        # Sort in-place by 'score' attribute (descending)
        group["documents"] = sorted(
            group_docs, key=lambda d: getattr(d, "score", 0), reverse=True)

    # # Merge diverse texts
    # merged_results: list[HeaderDocumentWithScore] = []
    # for group in grouped_results:
    #     all_texts = []
    #     for doc in group["documents"]:
    #         all_texts.extend(doc.metadata["texts"])
    #     diverse_texts = sample_diverse_texts(all_texts)
    #     text = "\n".join(all_texts)
    #     merged_results.append(HeaderDocumentWithScore(text=text, metadata={
    #         "header": group["headers"][0]
    #     }))

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Log performance
    logger.log(
        "main:",
        f"group_similar_headers: {execution_time:.2f}s, groups: {len(grouped_results)}",
        colors=["WHITE", "ORANGE"]
    )

    # Save results
    output_data = {
        "execution_time": f"{execution_time:.2f}s",
        "count": len(grouped_results),
        "results": grouped_results
    }
    save_file(output_data, f"{output_dir}/results.json")
    # save_file(merged_results, f"{output_dir}/merged_results.json")
