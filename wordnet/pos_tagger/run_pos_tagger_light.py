import json
import os
import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_jobs_ai_llm_python
from jet.logger import logger
from jet.wordnet.pos_tagger_light import POSTagger
from jet.wordnet.sentence import split_sentences
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    tagger = POSTagger()
    all_texts = []

    # texts = [
    #     "Dr. Jose Rizal is the only example of a genius in many fields who became the greatest hero of a nation",
    #     # "Which then spawned the short-lived First Philippine Republic.",
    #     # "It's more fun in Republic of the Congo."
    # ]
    texts = load_sample_jobs_ai_llm_python()[:5]
    sentences = [sent for text in texts for sent in split_sentences(text)]

    logger.info("Tagging Words:")
    for num, text in enumerate(tqdm(sentences, desc="Tagging texts"), start=1):
        pos_results = tagger.process_and_tag(text)
        tagged_text = tagger.format_tags(pos_results)
        merged_results = tagger.merge_multi_word_pos(pos_results)

        text_sub_dir = os.path.join(OUTPUT_DIR, f"{num}_result")

        logger.success(f"Tagged Text:\n{tagged_text}")
        save_file(tagged_text, f"{text_sub_dir}/tagged_text.txt")
        logger.success(
            f"POS Results:\n{json.dumps(pos_results, indent=2, ensure_ascii=False)}"
        )
        save_file(
            {"query": text, "count": len(pos_results), "results": pos_results},
            f"{text_sub_dir}/pos_results.json",
        )
        logger.success(
            f"Merged Results:\n{json.dumps(merged_results, indent=2, ensure_ascii=False)}"
        )
        save_file(
            {"query": text, "count": len(merged_results), "results": merged_results},
            f"{text_sub_dir}/merged_results.json",
        )

    docs_with_propn = tagger.filter_docs_by_pos_tags(texts, ["PROPN"])
    save_file(docs_with_propn, f"{OUTPUT_DIR}/docs_with_propn.json")
