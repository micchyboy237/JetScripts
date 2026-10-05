import os

from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_jobs
from jet.wordnet.n_grams import group_sentences_by_ngram

# Example usage
if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0],
    )

    texts = load_sample_jobs()

    false_start_ngrams_results = group_sentences_by_ngram(
        [text.lower() for text in texts], is_start_ngrams=False
    )
    save_file(
        false_start_ngrams_results, f"{output_dir}/false_start_ngrams_results.json"
    )

    true_start_ngrams_results = group_sentences_by_ngram(
        [text.lower() for text in texts], is_start_ngrams=True
    )
    save_file(true_start_ngrams_results, f"{output_dir}/true_start_ngrams_results.json")
