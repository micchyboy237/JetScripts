from jet.llm.classification import classify_sentence
from jet.transformers.formatters import format_json
from jet.file.utils import save_file
from jet.logger import logger
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Example usage
if __name__ == "__main__":
    sentences = [
        "She runs quickly, but he walks slowly.",
        "What a day!",
        "Close the door.",
        "If it rains, we stay home.",
    ]
    classification_results = []
    for s in sentences:
        classification = classify_sentence(s)
        classification_results.append({
            "sentence": s,
            "classification": classification
        })
    logger.success(format_json(classification_results))
    save_file(classification_results, f"{OUTPUT_DIR}/classification_results.json")