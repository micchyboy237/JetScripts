import os
import re
import shutil
from typing import List, Optional, TypedDict
from pathlib import Path
from jet.models.base import download_model_readmes
from jet.logger import logger

# MarkdownCodeExtractor and related definitions
CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\n(.*?)\n```"
FILE_PATH_PATTERN = r".*File Path: `?([^`]+)`?"
UNKNOWN = "text"


class CodeBlock(TypedDict):
    """Represents a code block with associated metadata."""
    code: str
    language: str
    file_path: Optional[str]
    extension: str
    preceding_text: str
    index: int


class MarkdownCodeExtractor:
    """Extracts code blocks from Markdown content."""
    LANGUAGE_EXTENSIONS = {
        "python": ".py",
        "javascript": ".js",
        "java": ".java",
        "html": ".html",
        "css": ".css",
        "cpp": ".cpp",
        "c": ".c",
        "ruby": ".rb",
        "go": ".go",
        "php": ".php",
        "cypher": ".cypher",
        "text": ".txt",
        "unknown": "",
    }

    def get_extension(self, language: str, file_path: Optional[str]) -> str:
        if file_path:
            return "." + file_path.split(".")[-1] if "." in file_path else ""
        return self.LANGUAGE_EXTENSIONS.get(language.lower(), "")

    def extract_code_blocks(self, markdown: str, with_text: bool = False) -> List[CodeBlock]:
        lines = markdown.strip().splitlines()
        code_blocks: List[CodeBlock] = []
        current_file_path: Optional[str] = None
        inside_code_block = False
        lang = None
        code_lines = []
        text_content = ""
        current_index = 0
        text_start_index = 0

        for line in lines:
            file_path_match = re.match(FILE_PATH_PATTERN, line)
            if file_path_match:
                current_file_path = file_path_match.group(1)
                if text_content and with_text:
                    code_blocks.append(
                        CodeBlock(
                            code=text_content.rstrip(),
                            language=UNKNOWN,
                            file_path=None,
                            extension=".txt",
                            preceding_text="",
                            index=text_start_index
                        )
                    )
                    text_content = ""
                continue
            if line.startswith("```"):
                if not inside_code_block:
                    if text_content and with_text:
                        code_blocks.append(
                            CodeBlock(
                                code=text_content.rstrip(),
                                language=UNKNOWN,
                                file_path=None,
                                extension=".txt",
                                preceding_text="",
                                index=text_start_index
                            )
                        )
                        text_content = ""
                    inside_code_block = True
                    lang = line.strip("`").strip() or UNKNOWN
                    code_lines = []
                    text_start_index = current_index
                else:
                    code_content = "\n".join(code_lines).rstrip()
                    if code_content:
                        extension = self.get_extension(lang, current_file_path)
                        code_blocks.append(
                            CodeBlock(
                                code=code_content,
                                language=lang,
                                file_path=current_file_path,
                                extension=extension,
                                preceding_text=text_content.rstrip() if with_text else "",
                                index=current_index
                            )
                        )
                    inside_code_block = False
                    current_file_path = None
                    text_content = ""
                    text_start_index = current_index + 1
                continue
            if inside_code_block:
                code_lines.append(line)
            else:
                text_content += line + "\n"
            current_index += 1

        if text_content and with_text:
            code_blocks.append(
                CodeBlock(
                    code=text_content.rstrip(),
                    language="text",
                    file_path=None,
                    extension=".txt",
                    preceding_text="",
                    index=text_start_index
                )
            )
        return code_blocks


def process_readme_files(input_dir: str, output_dir: str, include_text: bool = False):
    """Process all .md files in input_dir and extract code blocks."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    extractor = MarkdownCodeExtractor()

    # Find all .md files
    md_files = input_path.rglob("*.md")

    for md_file in md_files:
        model_name = md_file.stem.replace("_README", "")
        # Replace colons with underscores in model_name for paths
        sanitized_model_name = model_name.replace(":", "_")
        print(f"Processing {md_file.name}...")

        # Read Markdown content
        with open(md_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        # Extract code blocks
        code_blocks = extractor.extract_code_blocks(
            markdown_content, with_text=include_text)

        # Create model-specific directory using sanitized model_name
        model_dir = output_path / sanitized_model_name
        model_dir.mkdir(exist_ok=True)

        # Initialize language-specific counters
        lang_counters = {}

        # Save each code block to a file
        for block in code_blocks:
            lang = block["language"].lower()
            ext = block["extension"] or ".txt"
            code = block["code"]
            file_path = block["file_path"]
            preceding_text = block["preceding_text"]

            # Increment language-specific counter
            lang_counters[lang] = lang_counters.get(lang, 0) + 1
            counter = lang_counters[lang]

            # Create language-specific subdirectory
            lang_dir = model_dir / lang
            lang_dir.mkdir(exist_ok=True)

            # Use file_path if provided, else generate a name
            if file_path:
                filename = Path(file_path).name
            else:
                filename = f"{sanitized_model_name}_{counter}{ext}"

            # Sanitize filename to avoid invalid characters
            filename = "".join(
                c for c in filename if c.isalnum() or c in (".", "_", "-"))
            output_file = lang_dir / filename

            # Write code block to file with preceding text if applicable
            with open(output_file, "w", encoding="utf-8") as f:
                if preceding_text and lang != "text":
                    f.write(
                        f"# Preceding text:\n# {preceding_text.replace('\n', '\n# ')}\n\n")
                f.write(code)
            logger.success(f"  Saved ({lang}): {output_file}")

        if not code_blocks:
            logger.debug(f"  No code blocks found in {md_file.name}")


def main():
    overwrite = True
    hf_readme_download_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes"
    # Download MLX Model README.md files.
    download_model_readmes(hf_readme_download_dir, overwrite)

    # output_dir = os.path.join(os.path.dirname(__file__), "hf_extracted_code")
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes/code"

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    include_text = True  # Set to True to include non-code text
    process_readme_files(hf_readme_download_dir, output_dir, include_text)


if __name__ == "__main__":
    main()
