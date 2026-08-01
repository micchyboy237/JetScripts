import os
import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.wordnet.pos_tagger import POSTagger
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main_pos_tagger(texts, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    tagger = POSTagger()
    results = []
    for idx, text in enumerate(tqdm(texts, desc="Tagging texts")):
        pos_results = tagger.process_and_tag(text)
        results.append(
            {
                "index": idx,
                "text": text,
                "pos": pos_results,
            }
        )
        save_file(
            results,
            f"{output_dir}/pos_results.json",
            verbose=idx == 0 or idx == len(texts) - 1,
        )

    docs_with_propn = tagger.filter_docs_by_pos_tags(texts, ["PROPN"])
    save_file(docs_with_propn, f"{output_dir}/docs_with_propn.json")


if __name__ == "__main__":
    from jet.libs.bertopic.examples.mock import load_sample_jobs_ai_llm_python
    from jet.wordnet.sentence import split_sentences

    texts = load_sample_jobs_ai_llm_python()[:5]
    sentences = [sent for text in texts for sent in split_sentences(text)]

    main_pos_tagger(sentences, OUTPUT_DIR)
