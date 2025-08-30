import os
import hashlib
import re
from pathlib import Path
from typing import Coroutine, Literal, Optional
import codecs
import json
import shutil

from jet.code.python_code_extractor import remove_comments
from jet.code.rst_code_extractor import rst_to_code_blocks
from jet.logger import logger
from jet.utils.file import search_files

REPLACE_OLLAMA_MAP = {
    "llama-index-llms-openai": "llama-index-llms-ollama",
    "llama-index-embeddings-openai": "llama-index-embeddings-huggingface",
    "llama_index.llms.openai": "jet.llm.ollama.adapters.ollama_llama_index_llm_adapter",
    "llama_index.embeddings.openai": "llama_index.embeddings.huggingface",
    "llama_index.graph_stores.neo4j": "jet.vectors.adapters.neo4j_property_graph_adapter",
    "llama_index.postprocessor.cohere_rerank": "jet.models.embeddings.adapters.rerank_cross_encoder_llama_index_adapter",
    "langchain_openai": "jet.llm.ollama.base_langchain",
    "langchain_anthropic": "jet.llm.ollama.base_langchain",
    "langchain_ollama": "jet.llm.ollama.base_langchain",
    "OpenAIEmbeddings": "OllamaEmbeddings",
    "OpenAIEmbedding": "HuggingFaceEmbedding",
    "OpenAI": "OllamaFunctionCallingAdapter",
    "ChatOpenAI": "ChatOllama",
    "ChatAnthropic": "ChatOllama",
    "CohereRerank": "CrossEncoderRerank",
    "(\"gpt-4\")": "(model=\"llama3.2\")",
    "('gpt-4')": "(model=\"llama3.2\")",
    "(\"gpt-3.5\")": "(model=\"llama3.2\")",
    "(\'gpt-3.5\')": "(model=\"llama3.2\")",
    "openai:": "ollama:",
    "anthropic:": "ollama:",
    "gpt-4o-mini": "llama3.2",
    "claude-3-5-sonnet-latest": "llama3.2",
    "text-embedding-3-small": "mxbai-embed-large",
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
    "./data/paul_graham": "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data",
}

COMMENT_LINE_KEYWORDS = [
    "PDFReader",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "getpass",
    "import nest_asyncio",
    "nest_asyncio.apply()"
]


def generate_unique_function_name(line):
    unique_hash = hashlib.md5(line.encode('utf-8')).hexdigest()[:8]
    return f"run_async_code_{unique_hash}"


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
    has_keyword = any(keyword in line for keyword in COMMENT_LINE_KEYWORDS)
    updated_line = line
    if has_keyword and not line.strip().startswith('#'):
        updated_line = "# " + line
    return updated_line


def add_ollama_initializer_code(code: str):
    initializer_code = "from jet.llm.ollama.base import initialize_ollama_settings\ninitialize_ollama_settings()"
    return "\n\n".join([initializer_code, code])


def add_general_initializer_code(code: str):
    all_code = [code]
    setup_generated_dir_code = (
        "file_name = os.path.splitext(os.path.basename(__file__))[0]\n"
        "GENERATED_DIR = os.path.join(\"results\", file_name)\n"
        "os.makedirs(GENERATED_DIR, exist_ok=True)\n"
    ).strip()
    if "GENERATED_DIR" in code:
        all_code.insert(0, setup_generated_dir_code)
    import_code = "import os"
    if "import os" not in code and "os." in code:
        all_code.insert(0, import_code)
    return "\n\n".join(all_code)


def move_all_imports_on_top(code: str) -> str:
    import_pattern = re.compile(
        r'^\s*(from .+ import .+|import .+)', re.MULTILINE)
    lines = code.splitlines()
    imports = set()
    non_import_code = []
    in_import_block = False
    open_parens = 0
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        stripped_line = line.strip()
        if import_pattern.match(line):
            if not in_import_block:
                in_import_block = True
            current_import = stripped_line
            if '(' in line:
                open_parens += line.count('(')
            if ')' in line:
                open_parens -= line.count(')')
            while open_parens > 0 and line_idx + 1 < len(lines):
                line_idx += 1
                next_line = lines[line_idx]
                current_import += f"\n{next_line.strip()}"
                open_parens += next_line.count('(') - next_line.count(')')
            imports.add(current_import)
        else:
            non_import_code.append(line)
        line_idx += 1
    sorted_imports = sorted(list(imports))
    imports_block = '\n'.join(sorted_imports)
    non_import_block = '\n'.join(non_import_code)
    return imports_block + '\n\n' + non_import_block


def add_jet_logger(code: str):
    import_code = """\
import os
import shutil
from jet.logger import CustomLogger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")
    """.strip()
    log_done_code = 'logger.info("\\n\\n[DONE]", bright=True)'
    return "\n\n".join([import_code, code, log_done_code])


def replace_print_with_jet_logger(code: str):
    return code.replace("print(", "logger.debug(")


def update_code_with_ollama(code: str) -> str:
    code_lines = code.splitlines()
    updated_lines = []
    for line in code_lines:
        updated_line = replace_code_line(line)
        updated_line = replace_async_calls(updated_line)
        updated_line = comment_line(updated_line)
        updated_lines.append(updated_line)
    updated_code = "\n".join(updated_lines)
    updated_code = re.sub(
        r'from llama_index\.llms\.openai import OpenAI',
        'from jet.llm.ollama.base import Ollama',
        updated_code
    )
    updated_code = re.sub(r'Ollama\s*\((.*?)\)', r'Ollama(\1)', updated_code)
    updated_code = re.sub(
        r'model=["\']gpt-4[^"\']*["\']',
        'model="llama3.2", log_dir=f"{LOG_DIR}/chats"',
        updated_code
    )
    updated_code = re.sub(
        r'model=["\']gpt-3\.5[^"\']*["\']',
        'model="llama3.2", log_dir=f"{LOG_DIR}/chats"',
        updated_code
    )
    # Remove api_key arg
    updated_code = re.sub(
        r',\s*api_key\s*=\s*["\'][^"\']*["\']', '', updated_code
    )
    updated_code = re.sub(
        r'api_key\s*=\s*["\'][^"\']*["\']\s*,\s*', '', updated_code
    )
    updated_code = re.sub(
        r'api_key\s*=\s*["\'][^"\']*["\']', '', updated_code
    )

    updated_code = re.sub(
        r'OllamaEmbedding\s*\((.*?)\)',
        r'OllamaEmbedding(model="mxbai-embed-large")',
        updated_code
    )
    updated_code = re.sub(
        r'ChatOllama\s*\((.*?)\)',
        r'ChatOllama(model="llama3.2")',
        updated_code
    )
    updated_code = re.sub(
        r'OllamaEmbeddings\s*\((.*?)\)',
        r'OllamaEmbeddings(model="mxbai-embed-large")',
        updated_code
    )
    updated_code = re.sub(
        r'HuggingFaceEmbedding\s*\((.*?)\)',
        r'HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)',
        updated_code
    )
    updated_code = re.sub(
        r'docs0\s*=\s*loader\.load_data\(file=Path\(".*?/llama2\.pdf"\)\)',
        'from llama_index.core.readers.file.base import SimpleDirectoryReader\n'
        'docs0 = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()',
        updated_code
    )
    updated_code = re.sub(
        r'f+"{GENERATED_DIR}/',
        r'f"{GENERATED_DIR}/',
        updated_code
    )
    return updated_code


def read_notebook_file(file, with_markdown=False):
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
                if line.strip().startswith('#'):
                    continue
                if not line.endswith('\n'):
                    line += '\n'
                if line.strip().startswith(('%', '!')) and not line.strip().startswith('#'):
                    line = "# " + line
                code_lines.append(line)
            elif with_markdown:
                if not line.endswith('\n'):
                    line += '\n'
                code_lines.append(line)
        source_groups.append({
            "type": "text" if cell.get('cell_type') != "code" else "code",
            "code": "".join(code_lines).strip()
        })
    return source_groups


def read_markdown_file(file):
    from jet.code.markdown_code_extractor import MarkdownCodeExtractor
    if not (file.endswith('.md') or file.endswith('.mdx')):
        raise ValueError("File must have .md or .mdx extension")
    with open(file, 'r', encoding='utf-8') as f:
        source = f.read()
    extractor = MarkdownCodeExtractor()
    code_blocks = extractor.extract_code_blocks(source, with_text=True)
    source_groups = []
    for code_block in code_blocks:
        type = code_block["language"]
        lines = code_block["code"].splitlines()
        code_lines = []
        for line in lines:
            if type != 'text':
                if line.strip().startswith('#'):
                    continue
                if not line.endswith('\n'):
                    line += '\n'
            else:
                if not line.endswith('\n'):
                    line += '\n'
                if line.strip().startswith('pip install') and not line.strip().startswith('#'):
                    line = "# " + line
            code_lines.append(line)
        source_groups.append({
            "type": "code" if type != "text" else "text",
            "code": "".join(code_lines).strip()
        })
    return source_groups


def read_rst_file(file):
    if not file.endswith('.rst'):
        raise ValueError("File must have .rst extension")
    code_blocks = rst_to_code_blocks(file)
    source_groups = []
    for code_block in code_blocks:
        type = code_block["type"]
        lines = code_block["code"].splitlines()
        code_lines = []
        for line in lines:
            if type == 'python':
                if line.strip().startswith('#'):
                    continue
                if not line.endswith('\n'):
                    line += '\n'
            else:
                if not line.endswith('\n'):
                    line += '\n'
                if line.strip().startswith('pip install') and not line.strip().startswith('#'):
                    line = "# " + line
            code_lines.append(line)
        source_groups.append({
            "type": "text" if type != "code" else "code",
            "code": "".join(code_lines).strip()
        })
    return source_groups


async def run_async_wrapper(code: str) -> str:
    """Run async code in the current event loop and return the result as a string."""
    try:
        namespace = {}
        exec(code, namespace)
        for value in namespace.values():
            if isinstance(value, Coroutine):
                result = await value
                return f"logger.success(format_json({result}))"
        return ""
    except Exception as e:
        logger.error(f"Error executing async code: {e}")
        return f"logger.error('Error: {e}')"


def wrap_await_code(code: str) -> str:
    lines = code.splitlines()
    updated_lines = []
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].rstrip()
        leading_spaces = len(line) - len(line.lstrip())
        indent = " " * leading_spaces
        stripped_line = line.strip()

        # Handle async with blocks
        if stripped_line.startswith("async with"):
            variable = "result"
            async_block = [f"{indent}{stripped_line}"]
            line_idx += 1
            while line_idx < len(lines):
                next_line = lines[line_idx].rstrip()
                next_leading_spaces = len(next_line) - len(next_line.lstrip())
                if next_leading_spaces <= leading_spaces and next_line.strip():
                    break
                relative_indent = next_leading_spaces - leading_spaces
                if relative_indent < 0:
                    relative_indent = 0
                adjusted_indent = ' ' * (leading_spaces + 4 + relative_indent)
                async_block.append(f"{adjusted_indent}{next_line.lstrip()}")
                line_idx += 1
            async_block.append(
                f"{indent}logger.success(format_json({variable}))")
            updated_lines.extend(async_block)
            continue

        # Handle await statements (single-line or multi-line)
        if "await" in line:
            match = re.match(r'(.*?)\s*=\s*await', line)
            if match:
                variable = match.group(1).strip()
                if not variable:
                    updated_lines.append(line)
                    line_idx += 1
                    continue
                open_parens = stripped_line.count(
                    "(") - stripped_line.count(")")
                async_block = [f"{indent}{stripped_line}"]
                if open_parens > 0:  # Multi-line await
                    line_idx += 1
                    while line_idx < len(lines) and open_parens > 0:
                        next_line = lines[line_idx].rstrip()
                        next_leading_spaces = len(
                            next_line) - len(next_line.lstrip())
                        relative_indent = next_leading_spaces - leading_spaces
                        if relative_indent < 0:
                            relative_indent = 0
                        adjusted_indent = ' ' * \
                            (leading_spaces + 4 + relative_indent)
                        async_block.append(
                            f"{adjusted_indent}{next_line.lstrip()}")
                        open_parens += next_line.count("(") - \
                            next_line.count(")")
                        line_idx += 1
                else:  # Single-line await
                    line_idx += 1
                async_block.append(
                    f"{indent}logger.success(format_json({variable}))")
                updated_lines.extend(async_block)
                continue

        # Non-await lines
        updated_lines.append(line)
        line_idx += 1
    return "\n".join(updated_lines)


def wrap_triple_double_quoted_comments_in_log(code: str) -> str:
    pattern = r'"""\n?(.*?)\n?"""'
    matches = list(re.finditer(pattern, code, flags=re.DOTALL))
    updated_code = code
    offset = 0
    for match in matches:
        comment_content = match.group(1)
        lines = comment_content.splitlines()
        first_text_line = None
        for line in lines:
            if line.strip() and line.strip().startswith('#'):
                first_text_line = line.strip()
                break
        if not first_text_line:
            for line in lines:
                if line.strip() and line.strip()[0].isalnum():
                    first_text_line = line.strip()
                    break
        if first_text_line:
            replacement = f'logger.info("{first_text_line}")'
            start, end = match.start() + offset, match.end() + offset
            updated_code = updated_code[:end] + \
                '\n' + replacement + updated_code[end:]
            offset += len('\n' + replacement)
    return updated_code


def merge_consecutive_same_type(source_groups, separator="\n\n"):
    if not source_groups:
        return []
    source_groups = source_groups.copy()
    merged_groups = [source_groups[0]]
    for current in source_groups[1:]:
        last = merged_groups[-1]
        if last["type"] == current["type"]:
            last["code"] += separator + current["code"]
        else:
            merged_groups.append(current)
    return merged_groups


def scrape_code(
    input_base_dir: str,
    extensions: list[str],
    include_files: list[str] = [],
    exclude_files: list[str] = [],
    with_markdown: bool = True,
    with_ollama: bool = True,
    output_dir: Optional[str] = None,
    types: list[Literal['text', 'python']] = [],
):
    files = search_files(input_base_dir, extensions,
                         include_files, exclude_files)
    if include_files:
        files = [file for file in files if any(
            include.lower() in os.path.basename(file).lower() for include in include_files)]
    if exclude_files:
        files = [file for file in files if not any(
            exclude.lower() in os.path.basename(file).lower() for exclude in exclude_files)]
    logger.info(f"Found {len(files)} {extensions} files")
    results = []
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        try:
            if file.endswith('.ipynb'):
                source_groups = read_notebook_file(
                    file, with_markdown=with_markdown)
            elif file.endswith(('.md', '.mdx')):
                source_groups = read_markdown_file(file)
            elif file.endswith('.rst'):
                source_groups = read_rst_file(file)
            else:
                continue
            if types:
                source_groups = [
                    group for group in source_groups if group['type'] in types]
            merged_source_groups = merge_consecutive_same_type(source_groups)
            source_groups = merged_source_groups
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                subfolders = os.path.dirname(file).replace(
                    input_base_dir, '').strip('/')
                joined_dir = os.path.join(output_dir, subfolders)
                os.makedirs(joined_dir, exist_ok=True)
                output_code_path = os.path.join(joined_dir, f"{file_name}.py")
                for source_group in source_groups:
                    if source_group['type'] == 'text':
                        source_group['code'] = f'"""\n{source_group['code']}\n"""'
                    if source_group['type'] == 'text':
                        source_group['code'] = wrap_triple_double_quoted_comments_in_log(
                            source_group['code'])
                    if source_group['type'] == 'code':
                        source_group['code'] = wrap_await_code(
                            source_group['code'])
                source_code = "\n\n".join(group['code']
                                          for group in source_groups)
                if with_ollama:
                    source_code = update_code_with_ollama(source_code)
                    source_code = add_general_initializer_code(source_code)
                    source_code = add_jet_logger(source_code)
                    source_code = move_all_imports_on_top(source_code)
                    source_code = replace_print_with_jet_logger(source_code)
                if "format_json(" in source_code:
                    source_code = "from jet.transformers.formatters import format_json\n" + source_code
                if "HuggingFaceEmbedding" in source_code:
                    source_code = "from jet.models.config import MODELS_CACHE_DIR\n" + source_code
                if "await " in source_code:
                    source_code = "\n".join([
                        "async def main():",
                        *(f"    {line}" for line in source_code.splitlines()),
                        "",
                        "if __name__ == '__main__':",
                        "    import asyncio",
                        "    try:",
                        "        loop = asyncio.get_event_loop()",
                        "        if loop.is_running():",
                        "            loop.create_task(main())",
                        "        else:",
                        "            loop.run_until_complete(main())",
                        "    except RuntimeError:",
                        "        asyncio.run(main())",
                    ])
                with open(output_code_path, "w", encoding='utf-8') as f:
                    f.write(source_code)
                logger.log(
                    "Saved code:", output_code_path, colors=["GRAY", "BRIGHT_DEBUG"],
                )
                result = {
                    "data_file": file,
                    "code_file": output_code_path,
                    "code": source_code,
                }
                results.append(result)
        except Exception as e:
            logger.error(f"Failed to process file {file_name}: {e}")
    return results


def list_folders(paths: str | list[str]) -> list[str]:
    folders = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        if os.path.exists(path) and os.path.isdir(path):
            folders.extend(
                name for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name)) and not name.startswith(".")
            )
    return folders


def find_matching_repo_dir(input_base_dir: str, repo_base_dir: str | list[str], repo_dirs: list[str]) -> str | None:
    input_base_dir = os.path.abspath(input_base_dir)
    if isinstance(repo_base_dir, str):
        repo_base_dir = [repo_base_dir]
    repo_base_dir = sorted(repo_base_dir, key=len, reverse=True)
    repo_dirs = sorted(repo_dirs, key=len, reverse=True)
    for base_dir in repo_base_dir:
        for repo_dir in repo_dirs:
            repo_path = os.path.join(base_dir, repo_dir)
            if input_base_dir.startswith(repo_path):
                return repo_dir
    return None


def collect_files_and_dirs(input_base_dirs: list[str], extensions: list[str]) -> list[str]:
    """Collect all directories containing files with specified extensions recursively."""
    unique_dirs = set()
    for base_dir in input_base_dirs:
        base_path = Path(base_dir).resolve()
        if not base_path.exists():
            logger.warning(f"Directory does not exist: {base_dir}")
            continue
        if base_path.is_file():
            if any(base_path.suffix.lower() in ext.lower() for ext in extensions):
                unique_dirs.add(str(base_path.parent))
        elif base_path.is_dir():
            for ext in extensions:
                for file_path in base_path.rglob(f'*{ext}'):
                    unique_dirs.add(str(file_path.parent))
    return sorted(unique_dirs)


if __name__ == "__main__":
    repo_base_dir = [
        "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs",
        "/Users/jethroestrada/Desktop/External_Projects/AI/examples",
        "/Users/jethroestrada/Desktop/External_Projects/AI/lessons",
    ]
    repo_dirs = list_folders(repo_base_dir)
    input_base_dirs = [
        "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/llama_index",
    ]
    include_files = [
        # "agent_workflow_basic",
        # "agent_workflow_multi",
        # "agents_as_tools",
        "custom_multi_agent",
    ]
    exclude_files = [
        # "agent_workflow_research_assistant",
        # "agent_workflow_basic",
        # "Chatbot_SEC",
        # "multi_document_agents-v1",
        # "from_scratch_code_act_agent",
    ]
    extension_mappings = [
        {"ext": [".ipynb"], "output_base_dir": "converted-notebooks"},
        # {"ext": [".md"], "output_base_dir": "converted-markdown-docs"},
        {"ext": [".mdx"], "output_base_dir": "converted-markdown-extended"},
    ]
    all_extensions = [
        ext for mapping in extension_mappings for ext in mapping["ext"]]
    input_base_dirs = collect_files_and_dirs(input_base_dirs, all_extensions)
    print("Unique Containing Directories:", input_base_dirs)
    output_base_dir = os.path.dirname(__file__)
    for input_base_dir in input_base_dirs:
        logger.newline()
        logger.info(f"Processing: {input_base_dir}")
        matching_repo_dir = find_matching_repo_dir(
            input_base_dir, repo_base_dir, repo_dirs)
        logger.log("matching_repo_dir:", matching_repo_dir,
                   colors=["GRAY", "INFO"])
        if not matching_repo_dir:
            logger.error(f"No matching repo dir: \"{matching_repo_dir}\"")
            continue
        for ext_mapping in extension_mappings:
            extensions = ext_mapping["ext"]
            # Calculate relative path from matching_repo_dir to input_base_dir
            repo_path = os.path.join(
                [base for base in repo_base_dir if input_base_dir.startswith(
                    os.path.join(base, matching_repo_dir))][0],
                matching_repo_dir
            )
            relative_path = os.path.relpath(input_base_dir, repo_path)
            output_dir = os.path.join(
                output_base_dir,
                matching_repo_dir,
                ext_mapping["output_base_dir"],
                relative_path,
            )
            files = scrape_code(
                input_base_dir,
                extensions,
                include_files=include_files,
                exclude_files=exclude_files,
                with_markdown=True,
                with_ollama=True,
                output_dir=output_dir,
            )
            if files:
                logger.log(
                    "Saved", f"({len(files)})", "files to", output_dir,
                    colors=["WHITE", "SUCCESS", "WHITE", "BRIGHT_SUCCESS"],
                )
            else:
                logger.warning(f"No files processed in {input_base_dir}")
