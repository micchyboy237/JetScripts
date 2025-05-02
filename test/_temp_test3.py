

import os
import shutil
from jet.file.utils import load_file, save_file
from jet.wordnet.analyzers.classes.text_analyzer import TextAnalyzer


if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_html_myanimelist_net_Isekai/merged_docs.json"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/run_format_html"

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    docs: list[dict] = load_file(data_file)
    docs_with_text_analysis = []
    for doc in docs:
        text = doc["text"]

        # Initialize the TextAnalyzer
        analyzer = TextAnalyzer(language='english')

        # Analyze the text
        analysis = analyzer.analyze(text)

        docs_with_text_analysis.append({
            "text": text,
            "token_count": doc["token_count"],
            "analysis": analysis
        })

        save_file(docs_with_text_analysis,
                  f"{output_dir}/docs_with_text_analysis.json")
