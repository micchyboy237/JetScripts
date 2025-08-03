
from collections import Counter
import os
from jet.data.stratified_sampler import StratifiedSampler, get_ngrams
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

    all_ngrams = Counter()
    sentence_ngrams_dict = {}
    for sentence in texts:
        ngram_list = get_ngrams(sentence, n=2)
        all_ngrams.update(ngram_list)
        sentence_ngrams_dict[sentence] = ngram_list

    logger.gray(f"Results: ({len(all_ngrams)})")
    logger.success(format_json(all_ngrams))

    save_file(all_ngrams, f"{OUTPUT_DIR}/all_ngrams.json")
    save_file(sentence_ngrams_dict, f"{OUTPUT_DIR}/sentence_ngrams_dict.json")
