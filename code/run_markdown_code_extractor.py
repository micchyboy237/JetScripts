# /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/code/run_markdown_code_extractor.py

from jet.code.markdown_code_extractor import MarkdownCodeExtractor
from jet.file.utils import load_file, save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

md_content = load_file("/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/audio/audioFlux/docs/examples.md")

if __name__ == "__main__":
    extractor = MarkdownCodeExtractor()
    results = extractor.extract_code_blocks(md_content, with_text=True)
    save_file(results, f"{OUTPUT_DIR}/results.json")

    # Save each code block individually (with preceding text prepended as docstring for Python)
    python_counter = 0
    prev_text = None
    for result in results:
        if result["language"] == "text":
            prev_text = result["code"].strip()
            continue

        if result["language"] == "python":
            python_counter += 1
            counter = python_counter
        else:
            # For non-python code blocks, fall back to overall index if needed
            # But since we only have python in this case, this keeps flexibility
            counter = results.index(result) + 1

        content = result["code"]
        if result["language"] == "python" and prev_text is not None:
            if prev_text:
                content = f'"""\n{prev_text}\n"""\n\n{content}'
            prev_text = None

        lang_dir = result["extension"].lstrip(".")
        os.makedirs(f"{OUTPUT_DIR}/{lang_dir}", exist_ok=True)
        file_path = f"{OUTPUT_DIR}/{lang_dir}/{result['language']}_{counter}{result['extension']}"
        save_file(content, file_path)

    # Combine code blocks by language/extension
    # For python: prepend preceding text block as triple-quoted string
    combined_code = {}
    for result in results:
        key = (result["language"], result["extension"])
        combined_code.setdefault(key, []).append(result)

    for (language, ext), blocks in combined_code.items():
        parts = []
        prev_text = None
        for block in blocks:
            if block["language"] == "text":
                prev_text = block["code"].strip()
            else:
                if language == "python" and prev_text is not None:
                    if prev_text:
                        parts.append(f'"""\n{prev_text}\n"""\n')
                    prev_text = None
                parts.append(block["code"])

        # Append any trailing text if it precedes a non-python block (optional, kept for consistency)
        if language == "python" and prev_text is not None:
            if prev_text:
                parts.append(f'"""\n{prev_text}\n"""\n')

        combined_content = "\n\n".join(part for part in parts if part.strip())

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(OUTPUT_DIR, f"combined_{language}{ext}")
        save_file(combined_content, file_path)