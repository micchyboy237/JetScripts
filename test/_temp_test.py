from typing import List

from sudachipy import dictionary, tokenizer

# Initialize tokenizer once (global / singleton style)
_sudachi_tokenizer = None


def get_sudachi_tokenizer(split_mode: str = "B") -> tokenizer.Tokenizer:
    global _sudachi_tokenizer
    if _sudachi_tokenizer is None:
        # Loads core dictionary by default
        sudachi_dict = dictionary.Dictionary()
        splitmode_map = {
            "A": tokenizer.Tokenizer.SplitMode.A,
            "B": tokenizer.Tokenizer.SplitMode.B,
            "C": tokenizer.Tokenizer.SplitMode.C,
            "a": tokenizer.Tokenizer.SplitMode.A,
            "b": tokenizer.Tokenizer.SplitMode.B,
            "c": tokenizer.Tokenizer.SplitMode.C,
        }
        mode = splitmode_map.get(split_mode.upper(), tokenizer.Tokenizer.SplitMode.B)
        _sudachi_tokenizer = sudachi_dict.create(
            tokenizer.Tokenizer.SplitMode.B
        )  # default
        # We change mode per call → see below
    return _sudachi_tokenizer


def split_ja_phrases(
    text: str,
    mode: str = "B",  # "A", "B", "C"
    join_with_space: bool = False,
    filter_stopwords: bool = False,  # optional basic stopword removal
    return_surfaces_only: bool = True,
) -> List[str]:
    """
    Split Japanese text into phrase-like units using SudachiPy.

    Args:
        text:               Japanese input text
        mode:               Splitting granularity ("A"=short, "B"=medium, "C"=long)
        join_with_space:    Whether to join tokens with space (for Western-style output)
        filter_stopwords:   Remove very common functional tokens (basic heuristic)
        return_surfaces_only:
                            True  → return list of surface strings
                            False → return list of sudachipy.Morpheme objects

    Returns:
        List of phrase strings or Morpheme objects

    Example:
        >>> split_ja_phrases("私は昨日東京に行って寿司を食べました。")
        ['私', 'は', '昨日', '東京', 'に', '行って', '寿司', 'を', '食べました', '。']
        # with mode="C" → fewer splits on compounds
    """
    if not text.strip():
        return []

    tokenizer_obj = get_sudachi_tokenizer()
    sudachi_mode = {
        "A": tokenizer.Tokenizer.SplitMode.A,
        "B": tokenizer.Tokenizer.SplitMode.B,
        "C": tokenizer.Tokenizer.SplitMode.C,
    }.get(mode.upper(), tokenizer.Tokenizer.SplitMode.B)

    morphemes = tokenizer_obj.tokenize(text, sudachi_mode)

    if filter_stopwords:
        # Very rough heuristic — customize as needed
        stopwords = {
            "は",
            "が",
            "を",
            "に",
            "へ",
            "で",
            "と",
            "や",
            "の",
            "です",
            "ます",
            "。",
            "、",
        }
        morphemes = [m for m in morphemes if m.surface() not in stopwords]

    if return_surfaces_only:
        phrases = [m.surface() for m in morphemes]
        if join_with_space:
            return [" ".join(phrases)]
        else:
            return phrases
    else:
        return list(morphemes)
