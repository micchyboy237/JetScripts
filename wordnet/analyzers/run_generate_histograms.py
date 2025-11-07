import shutil
import os
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data#, load_sample_jobs
from jet.wordnet.analyzers.analyze_ngrams import generate_histograms


def main_generate_histograms(texts, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    results = generate_histograms(texts)
    save_file(results['most_common_start'], os.path.join(
        output_dir, 'most_common_start.json'),)
    save_file(results['least_common_start'], os.path.join(
        output_dir, 'least_common_start.json'),)
    save_file(results['most_common_any'], os.path.join(
        output_dir, 'most_common_any.json'),)
    save_file(results['least_common_any'], os.path.join(
        output_dir, 'least_common_any.json'),)


if __name__ == '__main__':
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    texts = load_sample_data()
    sub_output_dir = f"{output_dir}/anime"
    # texts = load_sample_jobs()
    # sub_output_dir = f"{output_dir}/jobs"

    main_generate_histograms(texts, sub_output_dir)
