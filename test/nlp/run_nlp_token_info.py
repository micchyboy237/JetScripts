import os
import shutil
import spacy
from typing import Any, Dict, List, TypedDict
from spacy.tokens import Token
from spacy.lexeme import Lexeme
from spacy.vocab import Vocab
from spacy.tokens import Doc, Span
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


class TokenInfo(TypedDict, total=False):
    text: str
    attributes: Dict[str, Any]


def safe_value(value: Any) -> Any:
    """
    Converts SpaCy and other non-serializable objects into safe, readable strings.
    """
    # Handle SpaCy-specific object types
    if isinstance(value, Token):
        return value.text
    if isinstance(value, Span):
        return value.text
    if isinstance(value, Doc):
        return value.text
    if isinstance(value, Lexeme):
        return value.text
    if isinstance(value, Vocab):
        return f"Vocab(size={len(value)})"

    # Handle iterables (children, lefts, etc.)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return [safe_value(v) for v in value]

    # Handle basic types
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Fallback for anything else
    return str(value)


def extract_token_info(text: str, model: str = "en_core_web_md") -> List[TokenInfo]:
    """
    Extracts all accessible Token attributes from a processed SpaCy Doc.
    Returns a list of TokenInfo dicts (one per token), all JSON-safe.
    """
    nlp = spacy.load(model)
    doc = nlp(text)

    # Collect non-dunder, non-callable Token attributes
    token_attrs = [
        attr for attr in dir(Token)
        if not attr.startswith("_")
        and not callable(getattr(Token, attr, None))
    ]

    token_infos: List[TokenInfo] = []

    for token in doc:
        info: Dict[str, Any] = {"text": token.text, "attributes": {}}
        for attr in sorted(token_attrs):
            try:
                raw_value = getattr(token, attr)
                info["attributes"][attr] = safe_value(raw_value)
            except Exception as e:
                info["attributes"][attr] = f"[Error: {e}]"
        token_infos.append(info)

    return token_infos


if __name__ == "__main__":
    sample_text = "Apple is looking at buying U.K. startup for $1 billion."
    results = extract_token_info(sample_text)

    for token_data in results:
        print(f"\nTOKEN: {token_data['text']}")
        print("-" * 40)
        for key, value in token_data["attributes"].items():
            print(f"{key:<20}: {value}")

    save_file(results, f"{OUTPUT_DIR}/results.json")