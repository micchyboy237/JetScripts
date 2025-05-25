import os
from jet.file.utils import save_file
from jet.wordnet.n_grams import count_ngrams
from jet.wordnet.n_grams.helpers import get_texts

# Example usage
if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    texts = get_texts()
    all_ngrams = count_ngrams([text.lower()
                              for text in texts], min_words=1, min_count=5)
    save_file(all_ngrams, f"{output_dir}/results.json")
