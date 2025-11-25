from typing import List, Optional
from transformers import pipeline, Pipeline
import logging

logger = logging.getLogger(__name__)


class JapaneseTranslator:
    """
    A reusable, lazily-initialized Japanese to English translator using
    Helsinki-NLP/opus-mt-ja-en (MarianMT-based, fast and lightweight).
    
    Designed for repeated use, supports batch translation, and is safe for
    both CPU and GPU environments (auto-detects available device).
    """

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-ja-en",
        device: Optional[int] = None,  # None = auto (cuda if available), -1 = CPU
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._translator: Optional[Pipeline] = None

    @property
    def translator(self) -> Pipeline:
        """Lazy-load the pipeline only when first needed."""
        if self._translator is None:
            logger.info(f"Loading translation model: {self.model_name}")
            self._translator = pipeline(
                "translation",
                model=self.model_name,
                device=self.device,        # handles CUDA if available
                max_length=self.max_length,
                truncation=True,
            )
            logger.info("Translation model loaded successfully")
        return self._translator

    def translate(self, text: str) -> str:
        """
        Translate a single Japanese string to English.
        
        Args:
            text: Japanese input text
            
        Returns:
            Translated English text (stripped of '>>en<<' prefix if present)
        """
        if not text.strip():
            return ""
            
        result = self.translator(text)[0]["translation_text"]
        # Some older MarianMT models prepend language tokens like ">>en<<"
        if result.startswith(">>en<<"):
            result = result[6:].lstrip()
        return result.strip()

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate multiple Japanese texts efficiently in batch.
        
        Args:
            texts: List of Japanese strings
            
        Returns:
            List of translated English strings in the same order
        """
        if not texts:
            return []
            
        cleaned_texts = [t.strip() for t in texts if t.strip()]
        if not cleaned_texts:
            return [""] * len(texts)

        results = self.translator(cleaned_texts)
        translations = [r["translation_text"] for r in results]
        
        # Clean language tokens
        cleaned = []
        for t in translations:
            if t.startswith(">>en<<"):
                t = t[6:].lstrip()
            cleaned.append(t.strip())
            
        # Reconstruct full list preserving empty inputs
        output = []
        clean_idx = 0
        for original in texts:
            if original.strip():
                output.append(cleaned[clean_idx])
                clean_idx += 1
            else:
                output.append("")
        return output

    def __call__(self, text: str) -> str:
        """Make the instance callable for convenience."""
        return self.translate(text)

if __name__ == "__main__":
    # Basic usage
    translator = JapaneseTranslator()

    # Single translation
    japanese_text = "今日はとても良い天気ですね。"
    english = translator.translate(japanese_text)
    print(english)
    # Output: "It's very nice weather today."

    # Or use callable syntax
    english = translator("こんにちは、世界！")
    print(english)
    # Output: "Hello, world!"

    # Batch translation (more efficient)
    sentences = [
        # "おはようございます。",
        # "私はソフトウェアエンジニアです。",
        # "",  # empty strings are preserved
        # "寿司が大好きです！", 

        "世界各国が水面下で",
    ]

    translations = translator.translate_batch(sentences)
    for ja, en in zip(sentences, translations):
        print(f"JA: {ja} → EN: {en}")

    # Output:
    # JA: おはようございます。 → EN: Good morning.
    # JA: 私はソフトウェアエンジニアです。 → EN: I am a software engineer.
    # JA:  → EN: 
    # JA: 寿司が大好きです！ → EN: I love sushi!
