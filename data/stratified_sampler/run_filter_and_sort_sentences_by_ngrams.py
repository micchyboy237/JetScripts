
import os
from jet.data.stratified_sampler import StratifiedSampler
from jet.wordnet.n_grams import filter_and_sort_sentences_by_ngrams
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

texts = [
    "Similar text 1",
    "Similar text 2",
    "Different writing 3",
    "Different writing 4",
]

if __name__ == "__main__":
    num_samples = 2
    sampler = StratifiedSampler(texts, num_samples=num_samples)
    results = filter_and_sort_sentences_by_ngrams(
        texts, n=1, top_n=2, is_start_ngrams=False)

    logger.gray(f"Results: ({len(results)})")
    logger.success(format_json(results))

    save_file(results, f"{OUTPUT_DIR}/results.json")
