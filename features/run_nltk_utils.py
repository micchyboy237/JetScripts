import os
from jet.features.nltk_utils import get_word_counts_lemmatized
from jet.file.utils import load_file, save_file

# Example usage
if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers: list[dict] = load_file(docs_file)
    docs = [header["text"] for header in headers]

    # Sample query (valid sentence)
    query = "List trending isekai reincarnation anime this year."

    docs_str = "\n\n".join(docs)

    results = get_word_counts_lemmatized(docs_str)

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    save_file(results, f"{output_dir}/word_counts_lemmatized.json")
