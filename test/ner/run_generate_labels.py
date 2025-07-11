import os
import shutil
from jet.file.utils import load_file, save_file
from jet.llm.mlx.templates.generate_labels import generate_labels
from jet.logger import logger

sample_texts = [
    "The Apollo 11 mission, launched on July 16, 1969, was the first manned moon landing, with astronauts Neil Armstrong and Buzz Aldrin setting foot on the lunar surface.",
    "Elon Musk founded Tesla, Inc. in 2003, which became a leading electric vehicle manufacturer based in Palo Alto, California."
]
output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)


def main_single_text():
    labels = generate_labels(sample_texts[0], model_path="qwen3-1.7b-4bit")
    save_file(labels, f"{output_dir}/gliner-labels.json")


def main_multiple_texts():
    labels_matrix = generate_labels(
        sample_texts, model_path="qwen3-1.7b-4bit")

    for labels in labels_matrix:
        logger.info(f"Generated labels: {labels}")
        for label in labels:
            logger.info(f"Processing label: {label}")

    save_file(labels_matrix, f"{output_dir}/gliner-labels-matrix.json")


if __name__ == "__main__":
    main_single_text()
    main_multiple_texts()
