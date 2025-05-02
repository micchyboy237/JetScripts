import nltk
from nltk.tokenize import sent_tokenize
import uuid

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)


def tokenize_text(text, skip_special_tokens=True, max_length=None):
    """
    Tokenizes the input text and returns a list of decoded token strings, merging sentences
    until the token count is under max_length using NLTK's sentence tokenizer to avoid
    losing information, along with metadata and the updated texts.

    Args:
        text (str): The input text to tokenize.
        skip_special_tokens (bool): Whether to skip special tokens in the output.
        max_length (int, optional): Maximum number of tokens to return. If None, returns all tokens.

    Returns:
        dict: Contains:
            - tokens: List of raw token strings.
            - decoded_tokens: List of decoded text versions of each token, excluding empty strings.
            - texts: The reconstructed text from the selected tokens.
            - metadata: Dict with total_tokens and is_truncated.
    """
    # Encode the text into token IDs
    token_ids = tokenizer.encode(text)
    total_tokens = len(token_ids)

    # If max_length is None or greater than total tokens, no truncation needed
    if max_length is None or max_length >= total_tokens:
        token_strings = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens)
        # Use batch_decode to decode all token IDs at once
        decoded_tokens = tokenizer.batch_decode(
            [[tid] for tid in token_ids], skip_special_tokens=skip_special_tokens
        )
        # Filter out empty strings
        decoded_tokens = [dt for dt in decoded_tokens if dt]
        # Reconstruct the updated texts from all token IDs
        updated_texts = tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens)
        return {
            "tokens": token_strings,
            "decoded_tokens": decoded_tokens,
            "texts": updated_texts,
            "metadata": {"total_tokens": total_tokens, "is_truncated": False}
        }

    # Get the decoded text to find sentence boundaries
    decoded_text = tokenizer.decode(
        token_ids, skip_special_tokens=skip_special_tokens)

    # Split text into sentences using NLTK
    sentences = sent_tokenize(decoded_text)

    # Re-tokenize sentences and accumulate tokens until max_length is reached
    selected_token_ids = []
    current_token_count = 0

    for sentence in sentences:
        sentence_token_ids = tokenizer.encode(sentence)
        if current_token_count + len(sentence_token_ids) <= max_length:
            selected_token_ids.extend(sentence_token_ids)
            current_token_count += len(sentence_token_ids)
        else:
            # Merge with the next sentence if possible to maximize content
            remaining_tokens = max_length - current_token_count
            if remaining_tokens > 0 and sentences.index(sentence) + 1 < len(sentences):
                next_sentence = sentences[sentences.index(sentence) + 1]
                merged_sentence = sentence + " " + next_sentence
                merged_token_ids = tokenizer.encode(merged_sentence)
                if len(merged_token_ids) <= max_length - current_token_count:
                    selected_token_ids.extend(merged_token_ids)
                    current_token_count += len(merged_token_ids)
                    # Mark as used
                    sentences[sentences.index(sentence) + 1] = ""
                else:
                    break
            else:
                break

    # Convert selected token IDs to token strings and decoded tokens
    token_strings = tokenizer.convert_ids_to_tokens(
        selected_token_ids, skip_special_tokens=skip_special_tokens)
    # Use batch_decode to decode all selected token IDs at once
    decoded_tokens = tokenizer.batch_decode(
        [[tid] for tid in selected_token_ids], skip_special_tokens=skip_special_tokens
    )
    # Filter out empty strings
    decoded_tokens = [dt for dt in decoded_tokens if dt]
    # Reconstruct the updated texts from selected token IDs
    updated_texts = tokenizer.decode(
        selected_token_ids, skip_special_tokens=skip_special_tokens)

    # Prepare metadata
    metadata = {
        "total_tokens": total_tokens,
        "is_truncated": len(selected_token_ids) < total_tokens
    }

    return {
        "tokens": token_strings,
        "decoded_tokens": decoded_tokens,
        "texts": updated_texts,
        "metadata": metadata
    }
