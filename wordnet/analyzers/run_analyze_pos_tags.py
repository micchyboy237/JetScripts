from jet.libs.bertopic.examples.mock import load_sample_data, load_sample_jobs
from jet.wordnet.analyzers.analyze_pos_tags import analyze_pos_tags
from jet.file.utils import save_file
import os
import shutil

def main_pos_tagger(texts, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Number of texts: {len(texts)}")
    save_file(texts, f"{output_dir}/texts.json")

    texts = [{"text": t, "lang": "en"} for t in texts]
    includes_pos = ['PROPN', 'NOUN', 'ADJ', 'VERB']
    top_n = 20

    analyze_pos_tags(texts, n=2, from_start=True,
                     words_only=True, top_n=top_n, includes_pos=includes_pos, output_dir=output_dir)
    analyze_pos_tags(texts, n=3, from_start=True,
                     words_only=True, top_n=top_n, includes_pos=includes_pos, output_dir=output_dir)
    excludes_pos = ['[BOS]', '[EOS]']
    analyze_pos_tags(texts, n=1, from_start=False,
                     words_only=True, top_n=top_n, includes_pos=includes_pos, output_dir=output_dir)
    analyze_pos_tags(texts, n=2, from_start=False,
                     words_only=True, top_n=top_n, includes_pos=includes_pos, output_dir=output_dir)
    analyze_pos_tags(texts, n=1, from_start=False,
                     excludes_pos=excludes_pos, top_n=top_n, includes_pos=includes_pos, output_dir=output_dir)
    analyze_pos_tags(texts, n=2, from_start=False,
                     excludes_pos=excludes_pos, top_n=top_n, includes_pos=includes_pos, output_dir=output_dir)

if __name__ == '__main__':
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    texts = load_sample_data()
    sub_output_dir = f"{output_dir}/anime"
    main_pos_tagger(texts, sub_output_dir)

    texts = load_sample_jobs()
    sub_output_dir = f"{output_dir}/jobs"
    main_pos_tagger(texts, sub_output_dir)
