import os
import json
from jet.file.utils import save_file
from jet.logger import logger
from jet.utils.text import format_sub_dir
from jet.wordnet.pos_tagger_light import POSTagger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


if __name__ == '__main__':
    tagger = POSTagger()
    all_texts = []

    texts = [
        "Dr. Jose Rizal is the only example of a genius in many fields who became the greatest hero of a nation",
        # "Which then spawned the short-lived First Philippine Republic.",
        # "It's more fun in Republic of the Congo."
    ]

    logger.info("Tagging Words:")
    for text in texts:
        pos_results = tagger.process_and_tag(text)
        tagged_text = tagger.format_tags(pos_results)
        merged_results = tagger.merge_multi_word_pos(pos_results)

        text_sub_dir = format_sub_dir(text)

        logger.success(f"Tagged Text:\n{tagged_text}")
        save_file(tagged_text, f"{text_sub_dir}/tagged_text.txt")
        logger.success(
            f"POS Results:\n{json.dumps(pos_results, indent=2, ensure_ascii=False)}")
        save_file({
            "query": text,
            "count": len(pos_results),
            "results": pos_results
        }, f"{text_sub_dir}/pos_results.json")
        logger.success(
            f"Merged Results:\n{json.dumps(merged_results, indent=2, ensure_ascii=False)}")
        save_file({
            "query": text,
            "count": len(merged_results),
            "results": merged_results
        }, f"{text_sub_dir}/merged_results.json")
