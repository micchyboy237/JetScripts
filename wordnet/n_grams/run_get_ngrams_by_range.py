import os

from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_jobs
from jet.wordnet.keywords.helpers import preprocess_texts
from jet.wordnet.n_grams import get_ngrams_by_range

# Example usage
if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0],
    )

    texts = load_sample_jobs()

    texts = preprocess_texts(texts)
    save_file(texts, f"{output_dir}/preprocessed_texts.json")

    results = list(
        get_ngrams_by_range(texts, min_words=1, max_words=2, count=2, show_count=True)
    )
    save_file(results, f"{output_dir}/results.json")
