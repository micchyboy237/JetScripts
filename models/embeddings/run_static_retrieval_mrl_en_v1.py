from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch
from tokenizers import Tokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download from the 🤗 Hub
model = SentenceTransformer(
    "sentence-transformers/static-retrieval-mrl-en-v1", device="cpu", backend="onnx"
)
tokenizer: Tokenizer = model.tokenizer

# Tokenize sentences


def tokenize_sentences(sentences: Union[str, List[str]]) -> dict:
    """Tokenize input sentences into token IDs and attention masks."""
    if isinstance(sentences, str):
        sentences = [sentences]
    logger.debug(f"Tokenizing {len(sentences)} sentences: {sentences[:50]}...")

    # Get padding token ID
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0  # Fallback to 0 if [PAD] is not in vocabulary
    logger.debug(f"Using pad_id: {pad_id}")

    # Configure tokenizer for padding and truncation
    tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=512)
    logger.debug(f"Tokenizer padding enabled with pad_id: {pad_id}")
    logger.debug(f"Tokenizer truncation enabled with max_length: 512")

    encodings = tokenizer.encode_batch(sentences)
    logger.debug(f"Encodings length: {len(encodings)}")
    input_ids = torch.tensor([enc.ids for enc in encodings], dtype=torch.long)
    attention_mask = torch.tensor(
        [enc.attention_mask for enc in encodings], dtype=torch.long)
    logger.debug(
        f"Input IDs shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}")
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# Detokenize token IDs


def detokenize_sentences(token_ids: torch.Tensor) -> List[str]:
    """Convert token IDs back to human-readable text."""
    logger.debug(f"Detokenizing token IDs with shape: {token_ids.shape}")
    detokenized = [tokenizer.decode(
        ids.tolist(), skip_special_tokens=True) for ids in token_ids]
    logger.debug(f"Detokenized sentences: {detokenized}")
    return detokenized

# Count tokens


def count_tokens(sentences: Union[str, List[str]]) -> List[int]:
    """Count the number of tokens in each sentence, excluding padding."""
    tokenized = tokenize_sentences(sentences)
    pad_id = tokenizer.token_to_id("[PAD]") or 0
    token_counts = [
        len([id for id in ids if id != pad_id])
        for ids in tokenized["input_ids"]
    ]
    logger.debug(f"Token counts: {token_counts}")
    return token_counts


# Input sentences
sentences = [
    'Gadofosveset-enhanced MR angiography of carotid arteries: does steady-state imaging improve accuracy of first-pass imaging?',
    'To evaluate the diagnostic accuracy of gadofosveset-enhanced magnetic resonance (MR) angiography in the assessment of carotid artery stenosis, with digital subtraction angiography (DSA) as the reference standard, and to determine the value of reading first-pass, steady-state, and "combined" (first-pass plus steady-state) MR angiograms.',
    'In a longitudinal study we investigated in vivo alterations of CVO during neuroinflammation, applying Gf-enhanced magnetic resonance imaging (MRI) in experimental autoimmune encephalomyelitis, an animal model of multiple sclerosis. SJL/J mice were monitored by Gadopentate dimeglumine- (Gd-DTPA) and Gf-enhanced MRI after adoptive transfer of proteolipid-protein-specific T cells. Mean Gf intensity ratios were calculated individually for different CVO and correlated to the clinical disease course. Subsequently, the tissue distribution of fluorescence-labeled Gf as well as the extent of cellular inflammation was assessed in corresponding histological slices.',
]

# Tokenize
tokenized = tokenize_sentences(sentences)
print("Token IDs shape:", tokenized["input_ids"].shape)

# Detokenize
detokenized = detokenize_sentences(tokenized["input_ids"])
print("Detokenized sentences:", detokenized)

# Count tokens
token_counts = count_tokens(sentences)
print("Token counts per sentence:", token_counts)

# Run inference
embeddings = model.encode(sentences)
print("Embeddings shape:", embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print("Similarities shape:", similarities.shape)
# [3, 3]
