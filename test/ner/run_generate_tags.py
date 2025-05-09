import os
import shutil
from jet.code.utils import ProcessedResult, process_markdown_file
from jet.file.utils import load_file, save_file
from jet.llm.mlx.templates.generate_tags import generate_tags


if __name__ == "__main__":
    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/docs/5_contextual_chunk_headers_rag.md"
    data: list[ProcessedResult] = process_markdown_file(md_file)
    texts: list[str] = [d["text"] for d in data]
    # Read arguments
    tags = generate_tags(texts)

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    save_file(tags, f"{output_dir}/tags.json")
