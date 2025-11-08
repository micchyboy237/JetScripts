import shutil
import os
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data, load_sample_jobs
from jet.wordnet.histogram import TextAnalysis


def main_text_analysis(texts, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    save_file(texts, os.path.join(output_dir, 'texts.json'))

    ta = TextAnalysis(texts)

    most_common_start = ta.generate_histogram(
        is_top=True,
        from_start=True,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 1), (2, 3)],
        top_n=200,
    )
    save_file(most_common_start, f"{output_dir}/most_common_start.json")

    least_common_start = ta.generate_histogram(
        is_top=False,
        from_start=True,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 1), (2, 3)],
        top_n=200,
    )
    save_file(least_common_start, f"{output_dir}/least_common_start.json")

    most_common_any = ta.generate_histogram(
        is_top=True,
        from_start=False,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 3), (4, 6)],
        top_n=200,
    )
    save_file(most_common_any, f"{output_dir}/most_common_any.json")

    least_common_any = ta.generate_histogram(
        is_top=False,
        from_start=False,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 3), (4, 6)],
        top_n=200,
    )
    save_file(least_common_any, f"{output_dir}/least_common_any.json")

    top_documents = ta.filter_top_documents_by_tfidf_and_collocations(
        ngram_range=(1, 2),
        # weight_tfidf=0.6,
        # weight_collocation=0.4,
        top_n=20,
        show_progress=True,
    )
    save_file(top_documents, f"{output_dir}/top_documents_n_1_2.json")

    top_documents = ta.filter_top_documents_by_tfidf_and_collocations(
        ngram_range=(3, 6),
        # weight_tfidf=0.6,
        # weight_collocation=0.4,
        top_n=20,
        show_progress=True,
    )
    save_file(top_documents, f"{output_dir}/top_documents_n_3_6.json")

if __name__ == '__main__':
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    texts = load_sample_data()
    sub_output_dir = f"{output_dir}/anime"
    main_text_analysis(texts, sub_output_dir)

    texts = load_sample_jobs()
    sub_output_dir = f"{output_dir}/jobs"
    main_text_analysis(texts, sub_output_dir)
