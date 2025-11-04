from jet.libs.bertopic.examples.mock import load_sample_data
from jet.wordnet.analyzers.analyze_pos_tags import analyze_pos_tags

if __name__ == '__main__':
    import os

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    texts = load_sample_data()

    analyze_pos_tags(texts, n=2, from_start=True,
                     words_only=True, output_dir=output_dir)
    analyze_pos_tags(texts, n=3, from_start=True,
                     words_only=True, output_dir=output_dir)
    excludes_pos = ['[BOS]', '[EOS]']
    analyze_pos_tags(texts, n=1, from_start=False,
                     words_only=True, output_dir=output_dir)
    analyze_pos_tags(texts, n=2, from_start=False,
                     words_only=True, output_dir=output_dir)
    analyze_pos_tags(texts, n=1, from_start=False,
                     excludes_pos=excludes_pos, output_dir=output_dir)
    analyze_pos_tags(texts, n=2, from_start=False,
                     excludes_pos=excludes_pos, output_dir=output_dir)
