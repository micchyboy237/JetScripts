# example_3_m2m100.py
# pip install transformers sentencepiece torch

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
from typing import Optional

class M2MTranslator:
    def __init__(self, model_name: str = "facebook/m2m100_418M", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def translate(self, text: str, src_lang: str = "ja", tgt_lang: str = "en") -> str:
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        forced_bos_token_id = self.tokenizer.get_lang_id(tgt_lang)
        generated = self.model.generate(**encoded, forced_bos_token_id=forced_bos_token_id, max_new_tokens=200)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

if __name__ == "__main__":
    tr = M2MTranslator()
    print(tr.translate("こんにちは、元気ですか？", src_lang="ja", tgt_lang="en"))
