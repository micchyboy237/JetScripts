import os
from jet.file.utils import save_file
from jet.wordnet.similarity import group_similar_texts


sample_texts = [
    "I love programming in Python.",
    "Python is my favorite programming language.",
    "The weather is great today.",
    "It's a sunny and beautiful day.",
    "I enjoy coding in Python.",
    "Machine learning is fascinating.",
    "Artificial Intelligence is evolving rapidly.",
    "Watch 2025 isekai anime on Netflix."
]
if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    grouped_texts = group_similar_texts(sample_texts, threshold=0.7)

    save_file({"count": len(grouped_texts), "results": grouped_texts},
              f"{output_dir}/grouped_texts.json")
