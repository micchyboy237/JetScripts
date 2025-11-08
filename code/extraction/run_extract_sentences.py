import os
import shutil
from jet.code.extraction import extract_sentences
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data, load_sample_jobs

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def main_extract_sentences(texts, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    save_file(texts, f"{output_dir}/texts.json")

    sentences = extract_sentences(texts, use_gpu=True, verbose=True)
    results = [{
        "index": idx,
        "sentence": sent,
    } for idx, sent in enumerate(sentences)]
    save_file(results, f"{output_dir}/sentences.json")

if __name__ == '__main__':
    texts = load_sample_data()
    sub_output_dir = f"{OUTPUT_DIR}/anime"
    main_extract_sentences(texts, sub_output_dir)

    texts = load_sample_jobs()
    sub_output_dir = f"{OUTPUT_DIR}/jobs"
    main_extract_sentences(texts, sub_output_dir)
