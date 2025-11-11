from jet.code.markdown_code_extractor import MarkdownCodeExtractor
from jet.file.utils import load_file, save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

md_content = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/context_engineering/00_COURSE/10_guides_zero_to_hero/02_expand_context.py")

if __name__ == "__main__":
    extractor = MarkdownCodeExtractor()
    results = extractor.extract_code_blocks(md_content)
    save_file(results, f"{OUTPUT_DIR}/results.json")

    # Save each code block individually, as before
    for idx, result in enumerate(results):
        save_file(
            result["code"],
            f"{OUTPUT_DIR}/{result['extension'].lstrip('.')}/{result['language']}_{idx + 1}{result['extension']}"
        )

    # Now, combine code blocks by language/extension and save as one file
    combined_code = {}
    for result in results:
        key = (result["language"], result["extension"])
        combined_code.setdefault(key, []).append(result["code"])

    for (language, ext), code_blocks in combined_code.items():
        combined_content = ("\n\n").join(code_blocks)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(OUTPUT_DIR, f"combined_{language}{ext}")
        save_file(combined_content, file_path)
