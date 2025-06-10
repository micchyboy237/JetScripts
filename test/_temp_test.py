from typing import List
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import onnxruntime as ort

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress ONNX Runtime warnings by setting log level to ERROR
ort.set_default_logger_severity(3)  # 3 = ERROR, suppresses WARNING (2)


class SentenceEncoder:
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize the SentenceTransformer model with ONNX backend.

        Args:
            model_name (str): Name of the model (e.g., 'all-MiniLM-L12-v2').
            device (str): Device to run the model on ('cpu' for M1 CPU).
        """
        logger.info(f"Loading model {model_name} on device {device}")
        try:
            self.model = SentenceTransformer(
                model_name, backend="onnx", device=device)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Encode a list of sentences into embeddings.

        Args:
            sentences (List[str]): List of sentences to encode.

        Returns:
            np.ndarray: Array of sentence embeddings.
        """
        logger.debug(f"Encoding {len(sentences)} sentences")
        try:
            embeddings = self.model.encode(sentences, convert_to_numpy=True)
            logger.info(f"Generated embeddings with shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding sentences: {str(e)}")
            raise


def main():
    # Example usage
    model_name = "all-MiniLM-L12-v2"
    sentences = [
        "This is an example sentence",
        "Each sentence is converted to an embedding"
    ]

    # Initialize encoder
    encoder = SentenceEncoder(model_name, device="cpu")

    # Encode sentences
    embeddings = encoder.encode_sentences(sentences)
    logger.info(f"Embeddings:\n{embeddings}")


if __name__ == "__main__":
    main()
