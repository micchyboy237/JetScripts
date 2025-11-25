# example_2_ctranslate2_fixed.py
import os
import ctranslate2
from transformers import AutoTokenizer

DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ja-en"
DEFAULT_CT2_MODEL_DIR = os.path.expanduser("~/.cache/hf_translation_models/ct2-opus-ja-en")

def translate_ctranslate2(text: str, device: str = "cpu") -> str:
    """
    Translate Japanese → English using a CTranslate2-converted Helsinki-NLP model.
    """
    # Use the regular tokenizer call (no deprecated prepare_seq2seq_batch)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TRANSLATION_MODEL)

    # Tokenize the source text → list of tokens (as strings)
    source = tokenizer.tokenize(text)                # → List[str]
    source = [source]                                 # → List[List[str]] (batch of 1)

    translator = ctranslate2.Translator(DEFAULT_CT2_MODEL_DIR, device=device)

    # translate_batch expects List[List[str]] for the source
    results = translator.translate_batch(
        source,
        # max_decoding_length=512,
        # beam_size=5,               # optional: better quality than greedy (beam_size=1)
        # replace_unknowns=True,     # recommended for Helsinki models
    )

    # results[0] is the first (and only) batch element
    # results[0].hypotheses[0] is the best hypothesis (list of tokens)
    output_tokens = results[0].hypotheses[0]

    # Convert tokens back to text (detokenize)
    translation = tokenizer.decode(
        tokenizer.convert_tokens_to_ids(output_tokens),
        skip_special_tokens=True,
    )

    return translation


if __name__ == "__main__":
    print(translate_ctranslate2("世界各国が水面下で熾烈な情報戦を繰り広げる時代 睨み合う二つの国 東の汚染"))
    # print(translate_ctranslate2("こんにちは、元気ですか？"))