import os
import shutil
from jet.file.utils import load_file, save_file
from jet.wordnet.similarity import group_similar_texts


if __name__ == '__main__':
    # Load documents
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    docs = load_file(docs_file)
    headers = [
        f"{(doc["metadata"].get("parent_header") or "").lstrip('#').strip()}\n{doc["metadata"]["header"].lstrip('#').strip()}".lower().strip() for doc in docs if doc["metadata"]["header_level"] > 1]

    grouped_headers = group_similar_texts(headers, threshold=0.5)

    save_file({"count": len(grouped_headers), "results": grouped_headers},
              f"{output_dir}/grouped_headers.json")
