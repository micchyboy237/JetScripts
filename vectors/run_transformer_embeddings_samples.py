import pytest
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from jet.logger import logger
from transformers import AutoTokenizer
import torch
import numpy as np

# Check for MPS availability (for M1 optimization)
device = torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")
logger.info(f"Using device: {device}")

# Load the model (optimized for M1 with MPS if available)
model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to(device)  # Move model to MPS or CPU

# Load the corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2')

# List of texts
texts = [
    "This is the first sample sentence.",
    "Another sentence to encode and count tokens.",
    "Short text."
]

# --- Batch Encoding ---
# Batch encode texts to get embeddings
embeddings = model.encode(
    texts, batch_size=8, convert_to_numpy=True, device=device)
logger.info(f"Batch Embeddings shape: {embeddings.shape}")

# Batch tokenize to get token counts (batch_encode_plus)
encoded_batch = tokenizer.batch_encode_plus(
    texts,
    add_special_tokens=True,
    return_tensors='pt',  # Return PyTorch tensors
    padding=True,         # Pad to the longest sequence
    truncation=True       # Truncate to model's max length
)

# Batch tokenize (tokenizer)
encoded_batch2 = tokenizer(
    texts,
    add_special_tokens=True,
    return_tensors=None   # Return lists of token IDs
)['input_ids']

# Calculate token counts
token_counts_batch = [len(input_ids)
                      for input_ids in encoded_batch['input_ids'].tolist()]
token_counts_batch2 = [len(input_ids) for input_ids in encoded_batch2]

# Decode batch results for batch_encode_plus
decoded_texts_batch = tokenizer.batch_decode(
    encoded_batch['input_ids'].tolist(),
    skip_special_tokens=True
)

# Log batch results for batch_encode_plus
logger.info("Encoded 1 (batch_encode_plus):")
for text, count, ids, decoded in zip(texts, token_counts_batch, encoded_batch['input_ids'].tolist(), decoded_texts_batch):
    logger.success(
        f"Text: {text}\n"
        f"Token count: {count}\n"
        f"Token IDs: {ids}\n"
        f"Batch Decoded: {decoded}\n"
    )

# Decode batch results for tokenizer
decoded_texts_batch2 = tokenizer.batch_decode(
    encoded_batch2,
    skip_special_tokens=True
)

# Log batch results for tokenizer
logger.info("Encoded 2 (tokenizer):")
for text, count, ids, decoded in zip(texts, token_counts_batch2, encoded_batch2, decoded_texts_batch2):
    logger.success(
        f"Text: {text}\n"
        f"Token count: {count}\n"
        f"Token IDs: {ids}\n"
        f"Batch Decoded: {decoded}\n"
    )

# --- Single Text Encoding ---
# Single text encoding for the first text in the list
single_text = texts[0]
single_embedding = model.encode(
    single_text, convert_to_numpy=True, device=device)
logger.info(f"Single Embedding shape: {single_embedding.shape}")

# Single text tokenize (batch_encode_plus equivalent for single text)
encoded_single = tokenizer(
    single_text,
    add_special_tokens=True,
    return_tensors='pt',
    padding=True,
    truncation=True
)

# Single text tokenize (tokenizer equivalent)
encoded_single2 = tokenizer(
    single_text,
    add_special_tokens=True,
    return_tensors=None
)['input_ids']

# Calculate token counts for single text
token_count_single = len(encoded_single['input_ids'][0].tolist())
token_count_single2 = len(encoded_single2)

# Decode single text results
decoded_single = tokenizer.decode(
    encoded_single['input_ids'][0].tolist(), skip_special_tokens=True)
decoded_single2 = tokenizer.decode(encoded_single2, skip_special_tokens=True)

# Log single text results
logger.info("Single Text Encoded (batch_encode_plus equivalent):")
logger.success(
    f"Text: {single_text}\n"
    f"Token count: {token_count_single}\n"
    f"Token IDs: {encoded_single['input_ids'][0].tolist()}\n"
    f"Decoded: {decoded_single}\n"
)

logger.info("Single Text Encoded (tokenizer):")
logger.success(
    f"Text: {single_text}\n"
    f"Token count: {token_count_single2}\n"
    f"Token IDs: {encoded_single2}\n"
    f"Decoded: {decoded_single2}\n"
)

# Optional: Compute cosine similarity between embeddings
similarity_matrix = cosine_similarity(embeddings)
logger.info("Cosine Similarity Matrix between texts:")
logger.success(f"{similarity_matrix}")

# --- Pytest Tests ---


def test_batch_encode_plus_token_count():
    expected = [len(tokenizer(text, add_special_tokens=True)
                    ['input_ids']) for text in texts]
    result = token_counts_batch
    assert result == expected, f"Expected token counts {expected}, but got {result}"


def test_batch_tokenizer_token_count():
    expected = [len(tokenizer(text, add_special_tokens=True)
                    ['input_ids']) for text in texts]
    result = token_counts_batch2
    assert result == expected, f"Expected token counts {expected}, but got {result}"


def test_single_encode_token_count():
    expected = len(
        tokenizer(single_text, add_special_tokens=True)['input_ids'])
    result = token_count_single
    assert result == expected, f"Expected token count {expected}, but got {result}"


def test_single_tokenizer_token_count():
    expected = len(
        tokenizer(single_text, add_special_tokens=True)['input_ids'])
    result = token_count_single2
    assert result == expected, f"Expected token count {expected}, but got {result}"


def test_batch_decode_batch_encode_plus():
    expected = texts
    result = decoded_texts_batch
    assert result == expected, f"Expected decoded texts {expected}, but got {result}"


def test_batch_decode_tokenizer():
    expected = texts
    result = decoded_texts_batch2
    assert result == expected, f"Expected decoded texts {expected}, but got {result}"


def test_single_decode_batch_encode_plus():
    expected = single_text
    result = decoded_single
    assert result == expected, f"Expected decoded text {expected}, but got {result}"


def test_single_decode_tokenizer():
    expected = single_text
    result = decoded_single2
    assert result == expected, f"Expected decoded text {expected}, but got {result}"


def test_embedding_shape():
    expected = (len(texts), model.get_sentence_embedding_dimension())
    result = embeddings.shape
    assert result == expected, f"Expected embedding shape {expected}, but got {result}"


def test_single_embedding_shape():
    expected = (model.get_sentence_embedding_dimension(),)
    result = single_embedding.shape
    assert result == expected, f"Expected single embedding shape {expected}, but got {result}"
