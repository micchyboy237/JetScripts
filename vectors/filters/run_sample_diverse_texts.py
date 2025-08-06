
import os
import shutil
from typing import List
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.data.sample_diverse_texts import sample_diverse_texts
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.models.model_types import EmbedModelType
from jet.models.tokenizer.base import count_tokens
from jet.transformers.formatters import format_json
from jet.wordnet.n_grams import count_ngrams
from jet.wordnet.text_chunker import chunk_texts_with_data

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main(texts: List[str], min_words: int):
    results: List[str] = sample_diverse_texts(texts, n=min_words)

    logger.gray(f"Results: ({len(results)})")
    logger.success(format_json(results))

    sub_dir = f"{OUTPUT_DIR}/n_{min_words}"
    all_ngrams = count_ngrams(texts, min_words=min_words)
    save_file(all_ngrams,
              f"{sub_dir}/all_ngrams.json")

    result_ngrams = count_ngrams(results, min_words=min_words)
    save_file(result_ngrams,
              f"{sub_dir}/result_ngrams.json")

    save_file(results, f"{sub_dir}/results.json")


if __name__ == "__main__":
    texts = [
        "Machine learning is a method of data analysis that automates model building.",
        "Deep learning, a subset of machine learning, uses neural networks for complex tasks.",
        "Supervised learning involves training models on labeled datasets.",
        "Unsupervised learning finds patterns in unlabeled data using clustering.",
        "Python is a popular programming language for machine learning development.",
        "Reinforcement learning optimizes decisions through trial and error.",
        "Data preprocessing is critical for effective machine learning models."
    ]

    main(texts, 1)
    main(texts, 2)
