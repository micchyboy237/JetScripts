import shutil
import os
from jet.file.utils import load_file, save_file
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
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_html_myanimelist_net_Isekai/headers.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    data: list[dict] = load_file(data_file)
    texts = [d["content"] for d in data]

    main_generate_histograms(texts, output_dir)
