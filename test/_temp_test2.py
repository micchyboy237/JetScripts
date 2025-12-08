import json
import ctranslate2
from transformers import AutoTokenizer
from typing import List, Any

QUANTIZED_MODEL_PATH = "/Users/jethroestrada/.cache/hf_ctranslate2_models/ja_en_ct2"

def translate_ja_to_en(
    text: str,
    model_path: str = QUANTIZED_MODEL_PATH,
    tokenizer_name: str = "Helsinki-NLP/opus-mt-ja-en",
    beam_size: int = 5,
    max_decoding_length: int = 512,  # Changed from max_length
    device: str = "cpu",  # 'cuda' for GPU
) -> str:
    """
    Translates Japanese text to English using a quantized OPUS-MT model.
    
    Args:
        text: Input Japanese sentence.
        model_path: Path to converted CTranslate2 model.
        tokenizer_name: Hugging Face tokenizer ID.
        beam_size: Beam search width (higher = better quality, slower).
        max_decoding_length: Max output tokens.  # Updated docstring
        device: 'cpu' or 'cuda'.
    
    Returns:
        Translated English text.
    
    Raises:
        RuntimeError: If model loading fails.
    """
    # Load tokenizer (reusable across calls)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load quantized translator
    translator = ctranslate2.Translator(model_path, device=device)
    
    # Tokenize input
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    source_batch = [source_tokens]  # Batch of 1
    
    # Translate
    results = translator.translate_batch(
        source_batch,
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,  # Changed from max_length
        return_scores=False,  # Set True for log-prob scores
    )
    
    # Decode output
    target_tokens = results[0].hypotheses[0]
    translated = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens))
    return translated.strip()  # Clean up

def batch_translate_ja_to_en(
    texts: List[str],
    model_path: str = QUANTIZED_MODEL_PATH,
    tokenizer_name: str = "Helsinki-NLP/opus-mt-ja-en",
    beam_size: int = 5,
    max_decoding_length: int = 512,
    device: str = "cpu",
    **kwargs: Any,
) -> List[str]:
    """
    Batch version of translate_ja_to_en with identical default arguments.
    
    Accepts the same translation options as translate_ja_to_en.
    Extra kwargs are forwarded to ctranslate2.Translator.translate_batch.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    translator = ctranslate2.Translator(model_path, device=device)

    source_batch = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in texts
    ]

    results = translator.translate_batch(
        source_batch,
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
        **kwargs,  # allow extra ctranslate2 options (e.g. return_scores=True)
    )

    translations = [
        tokenizer.decode(
            tokenizer.convert_tokens_to_ids(result.hypotheses[0])
        ).strip()
        for result in results
    ]
    return translations

if __name__ == "__main__":
    # Example usage (single)
    japanese_text = "おい、そんな一気に冷たいものを食べると腹を壊す"
    english_translation = translate_ja_to_en(japanese_text)
    print("=== Single Translation Result ===")
    print(f"Input: {japanese_text}")
    print(f"Output: {english_translation}")

    # Example usage (batch)
    japanese_text = [
        "おい、そんな一気に冷たいものを食べると腹を壊す",
        "昨日、友達と一緒に映画を見に行きました。",
        "日本は美しい国ですね！"
    ]
    english_translations = batch_translate_ja_to_en(japanese_text)
    print("=== Batch Translation Results ===")
    print(f"Input: {japanese_text}")
    print(f"Outputs:\n{json.dumps(english_translations, indent=2)}")