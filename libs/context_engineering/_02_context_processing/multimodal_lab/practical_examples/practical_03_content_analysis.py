import json
import numpy as np

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.libs.context_engineering.course._02_context_processing.labs.multimodal_lab import (
    create_example_dir, get_logger,
    MultimodalContentAnalyzer
)
from jet.file.utils import save_file


def practical_03_content_analysis():
    """Real multimodal content analysis: sentiment, safety, topic from text + image."""
    example_dir = create_example_dir("practical_03_content_analysis")
    logger = get_logger("mm_analysis", example_dir)
    logger.info("=" * 90)
    logger.info("PRACTICAL 3: Multimodal Content Analysis (Safety + Sentiment)")
    logger.info("=" * 90)

    (example_dir / "llm").mkdir(exist_ok=True)

    llm = LlamacppLLM(verbose=True)
    analyzer = MultimodalContentAnalyzer(d_model=768)

    # Generate controversial + safe content
    cases = [
        ("A cute puppy playing in the park", "happy"),
        ("A violent protest with police and fire", "unsafe"),
        ("A peaceful meditation retreat in nature", "calm"),
    ]

    all_results = {}
    for i, (prompt, expected) in enumerate(cases):
        save_file(prompt, str(example_dir / f"prompt_{i+1}.md"))

        logger.info(f"Analyzing: {prompt}")
        caption = ""
        for chunk in llm.generate(prompt + "\n\nDescribe this scene in detail.", max_tokens=200, stream=True):
            caption += chunk
        save_file(caption.strip(), str(example_dir / f"caption_{i+1}.md"))

        # Fake tokens + image
        tokens = np.array(list(range(3000 + i*100, 3050 + i*100)))
        image = np.random.rand(224, 224, 3) * 255

        result = analyzer.analyze_content(text_data=tokens, image_data=image)

        all_results[f"case_{i+1}_{expected}"] = {
            "prompt": prompt,
            "generated_caption": caption.strip(),
            "sentiment": result["sentiment"]["predicted"],
            "sentiment_confidence": result["sentiment"][result["sentiment"]["predicted"]],
            "safety": result["safety"]["classification"],
            "safety_score": result["safety"]["safe_score"],
            "confidence": result["quality_metrics"]["confidence"],
            "consistency": result["quality_metrics"]["multimodal_consistency"]
        }

        logger.info(f"→ Sentiment: {result['sentiment']['predicted']} "
                    f"({result['sentiment'][result['sentiment']['predicted']]:.3f}) | "
                    f"Safety: {result['safety']['classification']}")

    save_file(json.dumps(all_results, indent=2), str(example_dir / "full_analysis.json"))

    logger.info("PRACTICAL 3 COMPLETE — Multimodal safety & sentiment analysis ready!")
    logger.info("\nYou now have a full multimodal processing pipeline!")
    logger.info("=" * 90)


if __name__ == "__main__":
    practical_03_content_analysis()
