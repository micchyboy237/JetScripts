import shutil
from pathlib import Path

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.token_utils import get_tokenizer
from jet.code.splitter_markdown_utils import extract_md_header_contents
from jet.file.utils import load_file, save_file
from jet.logger import logger

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    markdown_text = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/smolagents/tools/generated/visit_webpage_tool/visit_webpage_tool_logs/call_0013/page.md"
    )

    tokenizer = get_tokenizer(LLM_MODEL)
    header_contents = extract_md_header_contents(
        markdown_text,
        min_tokens_per_chunk=64,
        max_tokens_per_chunk=128,
        model=LLM_MODEL,
    )
    # logger.success(format_json(all_nodes))
    logger.success(f"Headers ({len(header_contents)})")
    # logger.success(
    #     f"num_tokens_content: {[h['num_tokens_content'] for h in header_contents]}"
    # )
    # logger.success(
    #     f"num_tokens_merged_content: {[h['num_tokens_merged_content'] for h in header_contents]}"
    # )

    save_file(header_contents, f"{OUTPUT_DIR}/header_contents.json")


if __name__ == "__main__":
    main()
