import os
from jet.file.utils import load_file, save_file
from jet.logger.config import colorize_log
from jet.token.token_utils import split_headers
from jet.vectors.document_types import HeaderDocument
from jet.vectors.document_tagging import integrated_document_tagging


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    docs = [HeaderDocument(**doc) for doc in docs]
    chunked_docs = split_headers(docs)
    scraped_data = [
        {
            "header": doc["metadata"]["header"],
            "text": doc["content"]
        }
        for doc in chunked_docs
    ]
    query = "List all ongoing and upcoming isekai anime 2025."

    # scraped_data = [
    #     {"header": "Section 1",
    #         "text": "The CDC recommends 46 grams of protein daily for women aged 19-70."},
    #     {"header": "Section 2", "text": "Carbohydrates are the body's main energy source."},
    #     {"header": "Section 3",
    #         "text": "Protein intake varies based on activity level and age, and carbs fuel exercise."}
    # ]
    # query = "how much protein should a female eat"
    results = integrated_document_tagging(query, scraped_data, num_clusters=2)

    for r in results:
        print(f"Header: {r['header']}")
        print(f"Text: {r['text']}")
        print(
            f"Relevance Tag: {colorize_log(r['relevance_tag'], "BRIGHT_SUCCESS" if r['relevance_tag'] == "relevant" else "BRIGHT_ERROR")} (Similarity: {r['relevance_similarity']:.3f})")
        print("Topic Labels:", [
              f"{l['label']} (Similarity: {l['similarity']:.3f})" for l in r['topic_labels']])
        print()

    save_file(results, f"{output_dir}/results.json")
