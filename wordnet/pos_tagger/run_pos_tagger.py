import os
import shutil
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data, load_sample_jobs
from jet.wordnet.pos_tagger import POSTagger
from tqdm import tqdm

def main_pos_tagger(texts, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    tagger = POSTagger()
    results = []
    for idx, text in enumerate(tqdm(texts, desc="Tagging texts")):
        pos_results = tagger.process_and_tag(text)
        results.append({
            "index": idx,
            "text": text,
            "pos": pos_results,
        })
        save_file(results, f"{output_dir}/pos_results.json", verbose=idx == 0 or idx == len(texts) - 1)

if __name__ == '__main__':
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    texts = load_sample_data()
    sub_output_dir = f"{output_dir}/anime"
    main_pos_tagger(texts, sub_output_dir)

    texts = load_sample_jobs()
    sub_output_dir = f"{output_dir}/jobs"
    main_pos_tagger(texts, sub_output_dir)
