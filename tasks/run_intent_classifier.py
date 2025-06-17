from jet.logger import logger
# from jet.tasks.intent_classifier import classify_text
from jet.transformers.formatters import format_json
from typing import List, Dict, Union, Optional
import torch
from transformers import pipeline
from jet.logger import logger
from jet.transformers.formatters import format_json


def classify_text(
    text: Union[str, List[str]],
    model_name: str = "Falconsai/intent_classification",
    batch_size: Optional[int] = None
) -> List[Dict[str, Union[str, float]]]:
    """
    Classify text(s) using a Hugging Face transformer model with MPS support.

    Args:
        text: A single string or list of strings to classify.
        model_name: Name of the transformer model to use (default: Falconsai/intent_classification).
        batch_size: Batch size for processing multiple texts (optional, used only for list input).

    Returns:
        List of dictionaries containing classification results (label and score).
    """
    # Determine device (MPS if available, else CPU)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.gray(f"Using device: {device}")

    # Initialize pipeline
    classifier = pipeline(
        "text-classification",
        model=model_name,
        device=torch.device(device)
    )

    # Process input (single text or batch)
    if isinstance(text, str):
        result = classifier(text)
    else:
        result = classifier(text, batch_size=batch_size or len(text))

    return result


if __name__ == "__main__":

    text = "Your text to classify here."
    result = classify_text(text, batch_size=4)
    logger.gray("Result:")
    logger.success(format_json(result))
