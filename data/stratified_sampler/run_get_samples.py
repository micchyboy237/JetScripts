
import os
from jet.data.stratified_sampler import ProcessedData, StratifiedSampler, filter_and_sort_sentences_by_ngrams
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


texts = [
    "Machine learning is a method of data analysis that automates model building.",
    "Deep learning, a subset of machine learning, uses neural networks for complex tasks.",
    "Supervised learning involves training models on labeled datasets.",
    "Unsupervised learning finds patterns in unlabeled data using clustering.",
    "Python is a popular programming language for machine learning development.",
    "Reinforcement learning optimizes decisions through trial and error.",
    "Data preprocessing is critical for effective machine learning models."
]

texts2 = [
    ProcessedData(source="Global markets rise", target="Positive outlook",
                  category_values=["positive", 1, 5.0, True], score=0.9),
    ProcessedData(source="Tech stocks soar", target="Market boom",
                  category_values=["positive", 2, 4.5, True], score=0.8),
    ProcessedData(source="Economy slows down", target="Negative outlook",
                  category_values=["negative", 3, 3.0, False], score=0.4),
    ProcessedData(source="Markets face uncertainty", target="Cautious approach",
                  category_values=["neutral", 4, 4.0, False], score=0.6),
    ProcessedData(source="Stocks rebound quickly", target="Recovery",
                  category_values=["positive", 5, 4.8, True], score=0.7),
    ProcessedData(source="Financial crisis looms", target="Downturn",
                  category_values=["negative", 6, 2.5, False], score=0.3)
]


def main(texts):
    sampler = StratifiedSampler(texts, num_samples=None)
    diverse_samples = sampler.get_samples()

    logger.gray(f"Samples: ({len(diverse_samples)})")
    logger.success(format_json(diverse_samples))

    return diverse_samples


if __name__ == "__main__":
    results = main(texts)
    save_file(results, f"{OUTPUT_DIR}/diverse_texts.json")

    results = main(texts2)
    save_file(results, f"{OUTPUT_DIR}/diverse_texts_with_data.json")
