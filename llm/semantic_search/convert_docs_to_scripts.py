import re
import fnmatch
import os
import codecs
import json
import shutil
from typing import Literal, Optional
from jet.code.rst_code_extractor import rst_to_code_blocks
from jet.logger import logger
from jet.utils.file import search_files

REPLACE_OLLAMA_MAP = {
    "llama-index-llms-openai": "llama-index-llms-ollama",
    "llama-index-embeddings-openai": "llama-index-embeddings-ollama",
    "llama_index.llms.openai": "llama_index.llms.ollama",
    "llama_index.embeddings.openai": "llama_index.embeddings.ollama",
    "langchain_openai": "langchain_ollama",
    "langchain_anthropic": "langchain_ollama",
    "OpenAIEmbeddings": "OllamaEmbeddings",
    "OpenAIEmbedding": "OllamaEmbedding",
    "ChatOpenAI": "ChatOllama",
    "ChatAnthropic": "ChatOllama",
    "OpenAI": "Ollama",
    "(\"gpt-4\")": "(model=\"llama3.1\")",
    "('gpt-4')": "(model=\"llama3.1\")",
    "(\"gpt-3.5\")": "(model=\"llama3.2\")",
    "(\'gpt-3.5\')": "(model=\"llama3.2\")",
}

REPLACE_ASYNC_MAP = {
    "aquery": "query",
    "aevaluate": "evaluate",
    "aget_response": "get_response",
    "achat": "chat",
    "acomplete": "complete",
    "astream": "stream",
    "apredict": "predict",
}

REPLACE_PATHS_MAP = {
    "./data/paul_graham": "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume",
}

COMMENT_LINE_KEYWORDS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "getpass",
]


def replace_code_line(line: str):
    updated_line = line
    for old_line, new_line in {**REPLACE_OLLAMA_MAP, **REPLACE_PATHS_MAP}.items():
        if old_line in line:
            updated_line = updated_line.replace(old_line, new_line)
    return updated_line


def replace_async_calls(line: str):
    if not "await" in line and not "async for" in line:
        return line

    updated_line = line
    for old_line, new_line in REPLACE_ASYNC_MAP.items():
        if old_line in line:
            updated_line = updated_line.replace(old_line, new_line)
            updated_line = updated_line.replace("await ", "")
            updated_line = updated_line.replace("async ", "")
    return updated_line


def comment_line(line: str):
    has_keyword = any(
        keyword in line for keyword in COMMENT_LINE_KEYWORDS)
    updated_line = line
    if has_keyword:
        if not line.strip().startswith('#'):
            updated_line = "# " + line
    return updated_line


def add_ollama_initialier_code(code: str):
    initializer_code = "from jet.llm.ollama import initialize_ollama_settings\ninitialize_ollama_settings()"
    return "\n\n".join([
        initializer_code,
        code,
    ])


def add_jet_logger(code: str):
    import_code = "from jet.logger import logger\n"
    log_done_code = '\n\nlogger.info("\\n\\n[DONE]", bright=True)'
    return "".join([
        import_code,
        code,
        log_done_code,
    ])


def update_code_with_ollama(code: str) -> str:
    updated_code = code

    # Replace with mapping
    code_lines = code.splitlines()
    updated_lines = []
    for line in code_lines:
        updated_line = line
        updated_line = replace_code_line(updated_line)
        updated_line = replace_async_calls(updated_line)
        updated_line = comment_line(updated_line)
        updated_lines.append(updated_line)
    updated_code = "\n".join(updated_lines)

    """
    For llama-index
    """

    # Replace imports
    updated_code = re.sub(
        r'from llama_index\.llms\.openai import OpenAI',
        'from llama_index.llms.ollama import Ollama',
        updated_code
    )

    # Replace OpenAI(...) calls with Ollama(...)
    updated_code = re.sub(
        r'Ollama\s*\((.*?)\)',
        r'Ollama(\1)',
        updated_code
    )

    # Replace model="gpt-*" patterns correctly
    updated_code = re.sub(
        r'model=["\']gpt-4[^"\']*["\']',
        'model="llama3.1", request_timeout=300.0, context_window=4096',
        updated_code
    )
    updated_code = re.sub(
        r'model=["\']gpt-3\.5[^"\']*["\']',
        'model="llama3.2", request_timeout=300.0, context_window=4096',
        updated_code
    )

    # Replace OpenAIEmbedding(...) to OllamaEmbedding(...)
    updated_code = re.sub(
        r'OllamaEmbedding\s*\((.*?)\)',
        r'OllamaEmbedding(model_name="nomic-embed-text")',
        updated_code
    )

    """
    For langchain-core
    """

    updated_code = re.sub(
        r'ChatOllama\s*\((.*?)\)',
        r'ChatOllama(model="llama3.1")',
        updated_code
    )

    updated_code = re.sub(
        r'OllamaEmbeddings\s*\((.*?)\)',
        r'OllamaEmbeddings(model="nomic-embed-text")',
        updated_code
    )

    return updated_code


# Function to extract Python code cells from a .ipynb file
def read_notebook_file(file, with_markdown=False):
    # Check if the file ends correct extension
    if not file.endswith('.ipynb'):
        raise ValueError("File must have .ipynb extension")

    with codecs.open(file, 'r', encoding='utf-8') as f:
        source = f.read()

    source_dict = json.loads(source)
    cells = source_dict.get('cells', [])
    source_groups = []

    for cell in cells:
        code_lines = []
        for line in cell.get('source', []):
            if cell.get('cell_type') == 'code':
                # Remove commented lines
                if line.strip().startswith('#'):
                    continue

                # Add newline at the end if missing
                if not line.endswith('\n'):
                    line += '\n'

                # Comment out installation lines
                if line.strip().startswith('%') or line.strip().startswith('!'):
                    if not line.strip().startswith('#'):
                        line = "# " + line

                code_lines.append(line)

            elif with_markdown:
                if not line.strip().startswith('#'):
                    line = "# " + line
                if not line.endswith('\n'):
                    line += '\n'
                code_lines.append(line)
        source_groups.append({
            "type": cell.get('cell_type'),
            "code": "".join(code_lines).strip()
        })

    return source_groups


# Function to extract Python code blocks from a .md or .mdx file
def read_markdown_file(file):
    from jet.code import MarkdownCodeExtractor

    # Check if the file ends correct extension
    if not (file.endswith('.md') or file.endswith('.mdx')):
        raise ValueError("File must have .md or .mdx extension")

    with open(file, 'r') as f:
        source = f.read()

    extractor = MarkdownCodeExtractor()
    code_blocks = extractor.extract_code_blocks(source, with_text=True)

    source_groups = []

    for code_block in code_blocks:
        language = code_block["language"]
        lines = code_block["code"].splitlines()
        code_lines = []
        for line in lines:
            if language != 'text':
                # Remove commented lines
                if line.strip().startswith('#'):
                    continue

            else:
                # Comment out each line for non code block
                # if not line.strip().startswith('#'):
                #     line = "# " + line

                # Comment out installation lines
                # if line.strip().startswith('pip install'):
                #     if not line.strip().startswith('#'):
                #         line = "# " + line
                pass

            # Add newline at the end if missing
            if not line.endswith('\n'):
                line += '\n'

            code_lines.append(line)

        source_groups.append({
            "type": language,
            "code": "".join(code_lines).strip()
        })

    return source_groups


# Function to extract Python code blocks from a .rst file
def read_rst_file(file):
    # Check if the file ends correct extension
    if not (file.endswith('.rst')):
        raise ValueError("File must have .md or .mdx extension")

    code_blocks = rst_to_code_blocks(file)

    source_groups = []

    for code_block in code_blocks:
        type = code_block["type"]
        lines = code_block["code"].splitlines()
        code_lines = []
        for line in lines:
            if type == 'python':
                # Remove commented lines
                if line.strip().startswith('#'):
                    continue

                # Add newline at the end if missing
                if not line.endswith('\n'):
                    line += '\n'
            else:
                # Comment out each line for non code block
                if not line.strip().startswith('#'):
                    line = "# " + line

                # Add new line at the end
                if not line.endswith('\n'):
                    line += '\n'

                # Comment out installation lines
                if line.strip().startswith('pip install'):
                    if not line.strip().startswith('#'):
                        line = "# " + line

            code_lines.append(line)
        source_groups.append({
            "type": type,
            "code": "".join(code_lines).strip()
        })

    return source_groups


def scrape_code(
    input_base_dir: str,
    extensions: list[str],
    include_files: list[str] = [],
    exclude_files: list[str] = [],
    with_markdown: bool = True,
    with_ollama: bool = True,
    output_base_dir: Optional[str] = None,
    types: list[Literal['text', 'python']] = [],
):

    # Read files with any of the extensions in extensions recursively
    files = search_files(input_base_dir, extensions,
                         include_files, exclude_files)

    # Apply include_files filter
    if include_files:
        files = [
            file for file in files
            if any(include.lower() in file.lower() for include in include_files)
        ]

    # Apply exclude_files filter
    if exclude_files:
        files = [
            file for file in files
            if not any(exclude.lower() in file.lower() for exclude in exclude_files)
        ]

    logger.info(f"Found {len(files)} {extensions} files")

    # output_files = []
    results = []

    # Process each file and extract Python code
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]

        try:
            if file.endswith('.ipynb'):
                source_groups = read_notebook_file(
                    file, with_markdown=with_markdown)
            elif file.endswith('.md') or file.endswith('.mdx'):
                source_groups = read_markdown_file(file)
            elif file.endswith('.rst'):
                source_groups = read_rst_file(file)

            if types:
                source_groups = [
                    group for group in source_groups
                    if group['type'] in types
                ]

            if output_base_dir:
                # Create output base directory if it doesn't exist
                os.makedirs(output_base_dir, exist_ok=True)
                # Get subfolders
                subfolders = os.path.dirname(file).replace(input_base_dir, '')
                joined_dir = os.path.join(
                    output_base_dir, subfolders.strip('/'))
                os.makedirs(joined_dir, exist_ok=True)
                output_code = os.path.join(joined_dir, f"{file_name}.py")
                output_groups = os.path.join(joined_dir, f"{file_name}.json")

                for source_group in source_groups:
                    if source_group['type'] == 'text':
                        source_group['code'] = f'"""\n{
                            source_group['code']}\n"""'

                source_lines = [source_group['code']
                                for source_group in source_groups]
                source_code = "\n\n".join(source_lines)

                if with_ollama:
                    source_code = update_code_with_ollama(source_code)
                    # Add initializer code
                    # source_code = add_ollama_initialier_code(source_code)
                    # source_code = add_jet_logger(source_code)

                with open(output_code, "w") as f:
                    f.write(source_code)

                with open(output_groups, "w") as f:
                    json.dump(source_groups, f, indent=2)

                logger.debug(f"Saved code: {output_code}")
                logger.debug(f"Saved groups: {output_groups}")

                result = {
                    "data_file": file,
                    "code_file": output_code,
                    "blocks_file": output_groups,
                    "code": source_code,
                    "blocks": source_groups
                }
                results.append(result)

        except Exception as e:
            print(f"Failed to process file {file_name}: {e}")

    # return output_files
    return results


if __name__ == "__main__":
    data_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/llama_index/docs/docs/understanding/putting_it_all_together"
    data_extensions = [".ipynb", ".md", ".mdx", ".rst"]
    rag_dir = "generated/rag"

    shutil.rmtree(rag_dir, ignore_errors=True)

    include_files = []
    exclude_files = []

    results = scrape_code(
        data_dir,
        data_extensions,
        include_files=include_files,
        exclude_files=exclude_files,
        with_markdown=True,
        with_ollama=True,
        output_base_dir=rag_dir,
    )
