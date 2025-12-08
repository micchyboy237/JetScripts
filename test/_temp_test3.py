import numpy as np
from typing import List, Optional
from scipy.special import softmax  # For probability conversion
import ctranslate2

def convert_logits_to_numpy(
    result: ctranslate2.TranslationResult,
    device: Optional[str] = "cpu"
) -> List[List[np.ndarray]]:
    """
    Converts TranslationResult.logits (list[list[StorageView]]) to NumPy arrays.
    
    Args:
        result: Single TranslationResult from translate_batch.
        device: Move views to this device before conversion (e.g., "cpu").
    
    Returns:
        Nested list: [hypothesis_idx][position_idx] -> np.ndarray shape=(vocab_size,)
    
    Raises:
        ValueError: If logits not present or device mismatch.
    """
    if not hasattr(result, 'logits') or not result.logits:
        raise ValueError("Logits not returned; ensure return_logits=True in translate_batch.")
    
    numpy_logits: List[List[np.ndarray]] = []
    for hyp_idx, hyp_logits in enumerate(result.logits):
        numpy_hyp = []
        for pos_view in hyp_logits:
            # Move to target device if specified
            if device and pos_view.device != device:
                pos_view = pos_view.to_device(device)
            
            # Zero-copy conversion to NumPy
            pos_array = np.frombuffer(
                pos_view.data, dtype=pos_view.dtype
            ).reshape(pos_view.shape)
            numpy_hyp.append(pos_array)
        numpy_logits.append(numpy_hyp)
    
    return numpy_logits

def analyze_top_tokens(
    numpy_logits: List[np.ndarray],
    vocab: List[str],  # Model vocabulary
    top_k: int = 5
) -> List[List[tuple[str, float]]]:
    """
    Computes top-k tokens and probs (via softmax) for a list of position logits.
    
    Args:
        numpy_logits: Flat list of np.ndarray (one per position).
        vocab: List of vocabulary strings.
        top_k: Number of top tokens to return per position.
    
    Returns:
        List of [(token_str, prob), ...] per position.
    """
    top_tokens = []
    for pos_logits in numpy_logits:
        probs = softmax(pos_logits)  # Shape: (vocab_size,)
        top_indices = np.argpartition(probs, -top_k)[-top_k:]  # Efficient top-k
        top_probs = probs[top_indices]
        sorted_idx = np.argsort(-top_probs)  # Descending
        pos_top = [(vocab[idx], top_probs[sorted_idx[i]]) for i, idx in enumerate(top_indices[sorted_idx])]
        top_tokens.append(pos_top)
    return top_tokens

import ctranslate2
import numpy as np

QUANTIZED_MODEL_PATH = "/Users/jethroestrada/.cache/hf_ctranslate2_models/ja_en_ct2"

# Setup (example with en-de model)
translator = ctranslate2.Translator(QUANTIZED_MODEL_PATH, device="cpu")
source_tokens = [["Hello", "world"]]  # Batch of 1

# Translate with logits
results = translator.translate_batch(
    source_tokens,
    return_logits_vocab=True,
    num_hypotheses=3,  # 3 beams
    beam_size=5
)

# Process first result (batch idx 0)
result = results[0]
print(f"Hypotheses: {result.hypotheses}")
print(f"Scores: {result.scores}")

# Convert logits
numpy_logits_per_hyp = convert_logits_to_numpy(result)  # List[3 lists of StorageView -> np.ndarray]

# Example: Analyze top tokens for first hypothesis
first_hyp_logits = numpy_logits_per_hyp[0]  # List of seq_len arrays
# Assume vocab loaded elsewhere, e.g., from tokenizer
vocab = ["<pad>", "Hello", "world", ...]  # Full vocab list
top_per_pos = analyze_top_tokens(first_hyp_logits, vocab, top_k=3)

for pos, tops in enumerate(top_per_pos):
    print(f"Position {pos}: {tops}")  # e.g., [('Hallo', 0.45), ('Hello', 0.12), ...]