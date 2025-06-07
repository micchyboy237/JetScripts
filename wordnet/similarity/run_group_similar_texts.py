import os
from jet.file.utils import save_file
from jet.wordnet.similarity import group_similar_texts


# sample_texts = [
#     "I love programming in Python.",
#     "Python is my favorite programming language.",
#     "The weather is great today.",
#     "It's a sunny and beautiful day.",
#     "I enjoy coding in Python.",
#     "Machine learning is fascinating.",
#     "Artificial Intelligence is evolving rapidly.",
#     "Watch 2025 isekai anime on Netflix."
# ]
if __name__ == '__main__':
    import os
    from jet.file.utils import load_file, save_file

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    documents = [
        "\n".join([
            doc["metadata"].get("parent_header") or "",
            doc["metadata"]["header"],
            # doc["metadata"]["content"]
        ]).strip()
        for doc in docs
    ]

    grouped_similar_texts = group_similar_texts(documents, threshold=0.5)

    save_file({"count": len(grouped_similar_texts), "results": grouped_similar_texts},
              f"{output_dir}/grouped_similar_texts.json")
