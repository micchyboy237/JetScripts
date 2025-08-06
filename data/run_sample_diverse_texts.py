
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
from jet.wordnet.n_grams import get_most_common_ngrams
from jet.wordnet.text_chunker import chunk_texts_with_data

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


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

    all_common_ngrams = get_most_common_ngrams(texts, min_words=1)
    save_file(all_common_ngrams, f"{OUTPUT_DIR}/all_common_ngrams.json")

    results: List[str] = sample_diverse_texts(texts)

    logger.gray(f"Results: ({len(results)})")
    logger.success(format_json(results))

    result_common_ngrams = get_most_common_ngrams(results, min_words=1)
    save_file(result_common_ngrams, f"{OUTPUT_DIR}/result_common_ngrams.json")

    save_file(results, f"{OUTPUT_DIR}/results.json")
