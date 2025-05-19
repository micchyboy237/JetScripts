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

# Batch encode texts to get embeddings
embeddings = model.encode(
    texts, batch_size=8, convert_to_numpy=True, device=device)
logger.info(f"Embeddings shape: {embeddings.shape}")

# Batch tokenize to get token counts
encoded = tokenizer.batch_encode_plus(
    texts,
    add_special_tokens=True,
    return_tensors='pt',  # Return PyTorch tensors
    padding=True,         # Pad to the longest sequence
    truncation=True       # Truncate to model's max length
)

encoded2 = tokenizer(
    texts,
    add_special_tokens=True,
    return_tensors=None   # Return lists of token IDs
)['input_ids']

# Calculate token counts
token_counts = [len(input_ids) for input_ids in encoded['input_ids'].tolist()]
token_counts2 = [len(input_ids) for input_ids in encoded2]

# Use batch_decode for Encoded 1
decoded_texts = tokenizer.batch_decode(
    encoded['input_ids'].tolist(),
    skip_special_tokens=True
)

# Log results for Encoded 1
logger.info("Encoded 1 (batch_encode_plus):")
for text, count, ids, decoded in zip(texts, token_counts, encoded['input_ids'].tolist(), decoded_texts):
    logger.success(
        f"Text: {text}\n"
        f"Token count: {count}\n"
        f"Token IDs: {ids}\n"
        f"Batch Decoded: {decoded}\n"
    )

# Use batch_decode for Encoded 2
decoded_texts2 = tokenizer.batch_decode(
    encoded2,
    skip_special_tokens=True
)

# Log results for Encoded 2
logger.info("Encoded 2 (tokenizer):")
for text, count, ids, decoded in zip(texts, token_counts2, encoded2, decoded_texts2):
    logger.success(
        f"Text: {text}\n"
        f"Token count: {count}\n"
        f"Token IDs: {ids}\n"
        f"Batch Decoded: {decoded}\n"
    )

# Optional: Compute cosine similarity between embeddings
similarity_matrix = cosine_similarity(embeddings)
logger.info("Cosine Similarity Matrix between texts:")
logger.success(f"{similarity_matrix}")
