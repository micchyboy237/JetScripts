from jet.llm.classification import classify_sentence
from jet.file.utils import save_file
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
    for s in sentences:
        print(f"{s}: {classify_sentence(s)}")
    save_file(sentences, f"{OUTPUT_DIR}/classification_results.json")