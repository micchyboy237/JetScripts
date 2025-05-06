import os
import shutil
from jet.file.utils import load_file, save_file
from jet.llm.mlx.templates.generate_labels import generate_labels


if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_html_myanimelist_net_Isekai/headers.json"
    data: list[dict] = load_file(data_file)
    data: list[str] = [d["content"] for d in data]
    # Read arguments
    labels = generate_labels(data, max_labels=10)

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    save_file(labels, f"{output_dir}/gliner-labels.json")
