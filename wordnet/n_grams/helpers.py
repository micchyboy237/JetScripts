from jet.file.utils import load_file


def get_texts() -> list[str]:
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    headers: list[dict] = load_file(docs_file)
    texts = [header["text"] for header in headers]
    return texts
