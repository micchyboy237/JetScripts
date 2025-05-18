# scripts/merge_context_files.py

import os
from pathlib import Path
import shutil

from jet.code.splitter_markdown_utils import count_md_header_contents, get_md_header_contents
from jet.logger import logger
from jet.vectors.document_types import HeaderDocument
from jet.file.utils import load_file, save_file
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.token_utils import get_tokenizer_fn, count_tokens
from jet.token.token_utils import split_headers
from jet.utils.file import find_files_recursively
from jet.wordnet.analyzers.classes.text_analyzer import TextAnalyzer
from jet.wordnet.analyzers.text_analysis import analyze_readability


def merge_context_md_files(base_dir: Path, output_file: Path) -> None:
    """
    Recursively finds all 'context.md' files and writes their contents
    into a single markdown file.

    Args:
        base_dir (Path): Directory to start search from.
        output_file (Path): Output file path.
    """
    context_files = list(base_dir.rglob("context.md"))

    with output_file.open("w", encoding="utf-8") as out_f:
        for path in sorted(context_files):
            out_f.write(f"# {path.relative_to(base_dir)}\n\n")
            out_f.write(path.read_text(encoding="utf-8"))
            out_f.write("\n\n---\n\n")


if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    chunk_size = 225
    chunk_overlap = 40
    # Search for all Python files in current directory and subdirectories
    target_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    matched_files = find_files_recursively("context.md", target_dir)
    docs = []
    for file in matched_files:
        sub_dir = os.path.basename(os.path.dirname(file))
        final_output_dir = f"{output_dir}/{sub_dir}"

        doc: str = load_file(file, verbose=False)
        header_count = count_md_header_contents(doc)

        if header_count > 5:
            logger.info("Analyzing readability...")
            readability = analyze_readability(doc)
            save_file(readability, f"{final_output_dir}/readability.json")

            if readability["mltd_category"] == "high":
                docs.append(
                    {"file": file, "header_count": header_count, "content": doc})

    # top_doc = max(docs, key=lambda d: d["header_count"])
    # selected_docs = [top_doc]

    tokenizer = get_tokenizer_fn(model)

    # Initialize the TextAnalyzer
    analyzer = TextAnalyzer(language='english')

    for item in docs:
        logger.newline()

        file = item["file"]
        sub_dir = os.path.basename(os.path.dirname(file))
        final_output_dir = f"{output_dir}/{sub_dir}"

        content = item["content"]
        header_count = item["header_count"]
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        headers = get_md_header_contents(content, headers_to_split_on)
        header_docs = [
            HeaderDocument(
                text=header["content"],
                doc_index=i,
                header_level=header["header_level"],
                header=header["header"],
                parent_header=header["parent_header"],
            )
            for i, header in enumerate(headers)
        ]
        header_texts = [doc.text for doc in header_docs]
        header_tokens = tokenizer(header_texts)
        header_token_counts = [
            len(tokens) for tokens in header_tokens]
        headers_with_tokens = [
            {"tokens": len(header_tokens), "text": doc.text,
             "metadata": doc.metadata}
            for doc, tokens in zip(header_docs, header_token_counts)]
        save_file({
            "file": file,
            "header_count": header_count,
            "min_tokens": min(header_token_counts),
            "max_tokens": max(header_token_counts),
            "headers": headers_with_tokens
        }, f"{final_output_dir}/headers.json")

        # Analyze the text
        headers_text = "\n\n\n".join(header_texts)
        logger.info("Analyzing text...")
        analysis = analyzer.analyze(headers_text)
        save_file(analysis, f"{final_output_dir}/analysis.json")

        splitted_docs = split_headers(
            header_docs,
            model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokens=header_tokens,
            # tokenizer=tokenizer
        )
        splitted_token_counts: list[int] = count_tokens(
            model, [doc.text for doc in splitted_docs], prevent_total=True)

        chunks_with_tokens = [{"tokens": tokens, "text": doc.text, "metadata": doc.metadata}
                              for doc, tokens in zip(splitted_docs, splitted_token_counts)]

        save_file({
            "file": file,
            "header_count": header_count,
            "min_tokens": min(splitted_token_counts),
            "max_tokens": max(splitted_token_counts),
            "chunks": chunks_with_tokens
        }, f"{final_output_dir}/chunks.json")

        contexts = [doc.get_content() for doc in splitted_docs]
        save_file(contexts, f"{final_output_dir}/contexts.json")

        context = "\n\n\n".join(contexts)
        save_file(context, f"{final_output_dir}/context.md")

        # Run LLM
        query = "List trending isekai anime this year."


# if __name__ == "__main__":
#     import sys

#     try:
#         base_path = Path(sys.argv[1]).expanduser().resolve()
#         if not base_path.is_dir():
#             raise ValueError(f"Provided path is not a directory: {base_path}")

#         output_path = base_path / "searched_isekai_anime.md"
#         merge_context_md_files(base_path, output_path)
#         print(f"✅ Merged content written to: {output_path}")

#     except IndexError:
#         print("Usage: python scripts/merge_context_files.py <base_dir>")
#     except Exception as e:
#         print(f"❌ Error: {e}")
