from jet.logger import logger
from jet.file.utils import load_file, save_file
import logging
import multiprocessing as mp
import torch
from time import time
from typing import List, Dict, Any
from span_marker import SpanMarkerModel
import numpy as np
import os
print("Resolving imports...")

print("Done resolving imports...")

# Set environment variables for CPU usage and stability
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure multiprocessing start method
mp.set_start_method("fork", force=True)


class NERProcessor:
    def __init__(self, model_name: str):
        """Initialize the SpanMarker model on CPU."""
        logger.info(f"Loading model: {model_name}")
        torch.set_num_threads(4)
        self.device = torch.device("cpu")
        logger.debug(f"Using device: {self.device}")
        logger.debug(f"PyTorch threads: {torch.get_num_threads()}")
        logger.debug(f"Process ID: {os.getpid()}")
        self.model = SpanMarkerModel.from_pretrained(
            model_name).to(self.device)
        logger.debug(f"Model device: {next(self.model.parameters()).device}")
        logger.debug(
            f"NumPy BLAS config: {np.__config__.show(mode='dicts')['Build Dependencies']['blas']['name']}")

    def predict_batch(self, texts: List[str], batch_size=8, show_progress=False) -> List[List[Dict[str, Any]]]:
        """Process multiple texts in a batch for NER."""
        logger.info(
            f"Processing {len(texts)} texts in batch on {self.device} with batch_size={batch_size}")
        logger.debug(f"Process ID: {os.getpid()}")
        try:
            entities = self.model.predict(
                texts, batch_size=batch_size, show_progress_bar=show_progress)
            logger.debug(f"Completed prediction for {len(texts)} texts")
            return entities
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise


def main():
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs = load_file(docs_file)
    prompts = [doc["metadata"]["content"][:50] for doc in docs]

    # Initialize processor
    processor = NERProcessor(
        "tomaarsen/span-marker-bert-base-fewnerd-fine-super")

    # Run inference
    start_time = time()
    results = processor.predict_batch(prompts, show_progress=True)
    total_time = time() - start_time
    logger.info(f"Batch processing completed in {total_time:.2f} seconds")
    logger.info(
        f"Average time per text: {total_time/len(prompts):.2f} seconds")

    # Log results
    for text, entities in zip(prompts, results):
        logger.info(f"\nText: {text}\nEntities: {entities}")

    save_file(results, f"{output_dir}/results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise
