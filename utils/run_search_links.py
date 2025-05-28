
import os
from jet.llm.utils.link_searcher import search_links
from jet.file.utils import load_file, save_file

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"

    query = "Tips and links to 2025 online registration steps for TikTok live selling in the Philippines, or recent guidelines for online sellers on TikTok in the Philippines, or 2025 registration process for TikTok live selling in the Philippines."

    # Load JSON data
    docs = load_file(docs_file)
    print(f"Loaded JSON data {len(docs)} from: {docs_file}")
    links = [{"url": doc_link["url"], "text": doc_link["text"]}
             for doc in docs for doc_link in doc["metadata"]["links"]]
    save_file(links, f"{output_dir}/links.json")

    results = search_links(links, query, top_k=5, embedding_threshold=0.0)

    output_path = f"{output_dir}/results.json"
    save_file({
        "query": query,
        "results": results
    }, output_path)
