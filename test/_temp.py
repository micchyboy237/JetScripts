import re
from typing import List, Optional

from fast_bunkai import FastBunkai


def split_sentences_ja(
    text: str,
    punctuations: Optional[str] = "、…・",
) -> List[str]:
    if not text.strip():
        return []

    splitter = FastBunkai()
    chunks = list(splitter(text))  # First pass: respect 。！？ properly

    if not punctuations:
        return [s.strip() for s in chunks if s.strip()]

    # Pattern: split *after* each extra punctuation, keeping it with the left side
    extra_punc_escaped = re.escape(punctuations)
    pattern = f"(?<=[{extra_punc_escaped}])\\s*(?![{extra_punc_escaped}])"

    result = []

    for chunk in chunks:
        # If no extra punctuations in this chunk → keep as-is
        if not re.search(f"[{extra_punc_escaped}]", chunk):
            cleaned = chunk.strip()
            if cleaned:
                result.append(cleaned)
            continue

        # Split after extra punctuation (lookbehind ensures punctuation stays left)
        pieces = re.split(pattern, chunk)

        for piece in pieces:
            cleaned = piece.strip()
            if cleaned:
                result.append(cleaned)

    return result


text = "🎼世界各国が水面下で熾烈な情報戦を繰り広げる時代、睨み合う2つの国、東のオスタニア西のウスタリス、 戦争を企てるオスタニア政府用心の動向を探るべく、ウェスタリスはオペレーションストリックスを発動 作戦を担うス合ーデエージェント黄昏れ、 00の顔を使い分ける彼の任務は家族を作ること 父・ロイドフォージャー、精神科・正体・スパイ・コードネーム黄昏れ 母、ヨルフォージャー、市役所職員、正体・殺しやコードネーム茨姫 娘・アーニャフォージャー、正体・心を読むことができるエスパー 犬・ボンドフォージャー、正体・未来を予知できる超能力犬 物狩りのため、疑似家族を作り、互いに正体を隠した彼らのミッションは続く 。"
phrases = split_sentences_ja(text, punctuations="、…・！？")
print("\n".join([f"{p}" for num, p in enumerate(phrases, start=1)]))
