from jet.wordnet.analyzers.analyze_pos_tags import analyze_pos_tags

if __name__ == '__main__':
    from jet.file.utils import load_file
    import os

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/docs.json"

    docs: list[dict] = load_file(docs_file)
    texts = [{'text': doc["text"], 'lang': 'en'} for doc in docs["documents"]]

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
