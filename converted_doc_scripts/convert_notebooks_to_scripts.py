import hashlib
import re
import fnmatch
import os
import codecs
import json
import shutil
from typing import Literal, Optional
from jet.code.python_code_extractor import remove_comments
from jet.code.rst_code_extractor import rst_to_code_blocks
from jet.logger import logger
from jet.utils.file import search_files

REPLACE_OLLAMA_MAP = {
    "llama-index-llms-openai": "llama-index-llms-ollama",
    "llama-index-embeddings-openai": "llama-index-embeddings-ollama",
    "llama_index.llms.openai": "jet.llm.ollama.base",
    "llama_index.embeddings.openai": "jet.llm.ollama.base",
    "langchain_openai": "jet.llm.ollama.base_langchain",
    "langchain_anthropic": "jet.llm.ollama.base_langchain",
    "langchain_ollama": "jet.llm.ollama.base_langchain",
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
    # Generate a unique hash based on the line content
    # Taking the first 8 characters for brevity
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
    has_keyword = any(
        keyword in line for keyword in COMMENT_LINE_KEYWORDS)
    updated_line = line
    if has_keyword:
        if not line.strip().startswith('#'):
            updated_line = "# " + line
    return updated_line


def add_ollama_initializer_code(code: str):
    initializer_code = "from jet.llm.ollama.base import initialize_ollama_settings\ninitialize_ollama_settings()"
    return "\n\n".join([
        initializer_code,
        code,
    ])


def add_general_initializer_code(code: str):
    all_code = [
        code,
    ]

    setup_generated_dir_code = (
        "file_name = os.path.splitext(os.path.basename(__file__))[0]\n"
        "GENERATED_DIR = os.path.join(\"results\", file_name)\n"
        "os.makedirs(GENERATED_DIR, exist_ok=True)\n"
    ).strip()
    if "GENERATED_DIR" in code:
        all_code.insert(0, setup_generated_dir_code)
        code = "\n\n".join(all_code)

    import_code = "import os"
    if "import os" not in code and "os." in code:
        all_code.insert(0, import_code)
        code = "\n\n".join(all_code)

    return code


def move_all_imports_on_top(code: str) -> str:
    # Regex pattern to identify import statements, including multi-line imports
    import_pattern = re.compile(
        r'^\s*(from .+ import .+|import .+)', re.MULTILINE)

    lines = code.splitlines()
    imports = []
    non_import_code = []
    in_import_block = False
    open_parens = 0

    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]

        # Check for the start of an import block
        if import_pattern.match(line):
            if not in_import_block:
                in_import_block = True
            imports.append(line)

            # Handle multi-line imports with parentheses
            if '(' in line:
                open_parens += line.count('(')
            if ')' in line:
                open_parens -= line.count(')')

            # Continue adding lines until we are out of the parentheses block
            while open_parens > 0 and line_idx + 1 < len(lines):
                line_idx += 1
                next_line = lines[line_idx]
                imports[-1] += f"\n{next_line}"
                open_parens += next_line.count('(') - next_line.count(')')

            # Ensure imports are separated by new lines
            # Remove any trailing spaces after concatenating multi-line imports
            imports[-1] = imports[-1]

        else:
            # Collect non-import code
            non_import_code.append(line)

        line_idx += 1

    # Join imports with a newline for correct separation and return the final code
    imports_block = '\n'.join(imports)
    non_import_block = '\n'.join(non_import_code)

    return imports_block + '\n\n' + non_import_block


def add_jet_logger(code: str):
    import_code = """from jet.logger import CustomLogger
    
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")
    """.strip()
    log_done_code = 'logger.info("\\n\\n[DONE]", bright=True)'
    return "\n\n".join([
        import_code,
        code,
        log_done_code,
    ])


def replace_print_with_jet_logger(code: str):
    return code.replace("print(", "logger.debug(")


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
        'from jet.llm.ollama.base import Ollama',
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
        r'OllamaEmbedding(model_name="mxbai-embed-large")',
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
        r'OllamaEmbeddings(model="mxbai-embed-large")',
        updated_code
    )

    # Replace the line that loads data using `loader.load_data` with the updated implementation
    # that uses `SimpleDirectoryReader` and adds the appropriate import statement.
    updated_code = re.sub(
        r'docs0\s*=\s*loader\.load_data\(file=Path\(".*?/llama2\.pdf"\)\)',
        'from llama_index.core.readers.file.base import SimpleDirectoryReader\n'
        'docs0 = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()',
        updated_code
    )

    # Replace all occurrences of "data/ with f"{GENERATED_DIR}/" to use the dynamic directory variable.
    updated_code = re.sub(
        r'"data/',  # Matches the literal string "data/
        # Replaces it with the formatted string f"{GENERATED_DIR}/
        r'f"{GENERATED_DIR}/',
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
                # if not line.strip().startswith('#'):
                #     line = "# " + line
                if not line.endswith('\n'):
                    line += '\n'
                code_lines.append(line)
        source_groups.append({
            "type": "text" if cell.get('cell_type') != "code" else "code",
            "code": "".join(code_lines).strip()
        })

    return source_groups


# Function to extract Python code blocks from a .md or .mdx file
def read_markdown_file(file):
    from jet.code.markdown_code_extractor import MarkdownCodeExtractor

    # Check if the file ends correct extension
    if not (file.endswith('.md') or file.endswith('.mdx')):
        raise ValueError("File must have .md or .mdx extension")

    with open(file, 'r') as f:
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
                # Remove commented lines
                if line.strip().startswith('#'):
                    continue

                # Add newline at the end if missing
                if not line.endswith('\n'):
                    line += '\n'
            else:
                # Comment out each line for non code block
                # if not line.strip().startswith('#'):
                #     line = "# " + line

                # Add new line at the end
                if not line.endswith('\n'):
                    line += '\n'

                # Comment out installation lines
                if line.strip().startswith('pip install'):
                    if not line.strip().startswith('#'):
                        line = "# " + line

            code_lines.append(line)
        source_groups.append({
            "type": "code" if type != "text" else "text",
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
                # if not line.strip().startswith('#'):
                #     line = "# " + line

                # Add new line at the end
                if not line.endswith('\n'):
                    line += '\n'

                # Comment out installation lines
                if line.strip().startswith('pip install'):
                    if not line.strip().startswith('#'):
                        line = "# " + line

            code_lines.append(line)
        source_groups.append({
            "type": "text" if type != "code" else "code",
            "code": "".join(code_lines).strip()
        })

    return source_groups


def wrap_await_code_singleline_args(code: str) -> str:
    lines = code.splitlines()

    for line_idx, line in enumerate(lines):
        if "await" in line and line.strip().endswith("("):
            continue

        # Use regex to capture everything before "= await"
        match = re.match(r'(.*?)(?=\s*= await)', line)

        text_before_await = ""
        if match:
            text_before_await = match.group(1)
            print(f"Line {line_idx}: {text_before_await}")

        await_leading_spaces = len(
            line) - len(line.lstrip())
        if not await_leading_spaces:
            function_name = generate_unique_function_name(line)

            # Create the wrapped async code with a unique function name
            async_wrapped_code = "\n".join([
                f"async def {function_name}():",
                f"  {line}",
                f"  return {text_before_await}",
                f"\n{text_before_await} = asyncio.run({
                    function_name}())",
                f"logger.success(format_json({
                    text_before_await}))",
            ])

            lines[line_idx] = async_wrapped_code
            logger.debug(async_wrapped_code)
    return "\n".join(lines)


def wrap_await_code_multiline_args(code: str) -> str:
    """Wrap lines containing 'await' in standalone async functions, handling multiline calls."""
    lines = code.splitlines()
    updated_lines = []
    line_idx = 0

    while line_idx < len(lines):
        line = lines[line_idx]

        if "await" in line and line.strip().endswith("("):
            match = re.match(r'(.*?)\s*=\s*await', line)
            if match:
                variable = match.group(1).strip()
                leading_spaces = len(line) - len(line.lstrip())
                async_fn_name = f"async_func_{line_idx}"

                # Collect multiline call
                async_block = [
                    f"{' ' * leading_spaces}async def {async_fn_name}():"]
                async_block.append(
                    f"{' ' * (leading_spaces + 4)}{line.strip()}")

                open_parens = 1
                line_idx += 1
                while line_idx < len(lines) and open_parens > 0:
                    next_line = lines[line_idx]
                    async_block.append(
                        f"{' ' * (leading_spaces + 4)}{next_line.strip()}")
                    open_parens += next_line.count("(") - \
                        next_line.count(")")
                    line_idx += 1

                async_block.append(
                    f"{' ' * (leading_spaces + 4)}return {variable}")
                async_block.append(
                    f"{' ' * leading_spaces}{variable} = asyncio.run({async_fn_name}())")
                async_block.append(
                    f"{' ' * leading_spaces}logger.success(format_json({variable}))")

                updated_lines.extend(async_block)
            else:
                updated_lines.append(line)
                line_idx += 1
        elif "await" in line:
            # Use regex to capture everything before "= await"
            match = re.match(r'(.*?)(?=\s*= await)', line)

            text_before_await = ""
            if match:
                text_before_await = match.group(1)
                print(f"Line {line_idx}: {text_before_await}")

            await_leading_spaces = len(
                line) - len(line.lstrip())
            if not await_leading_spaces:
                function_name = generate_unique_function_name(line)

                # Create the wrapped async code with a unique function name
                async_wrapped_code = "\n".join([
                    f"async def {function_name}():",
                    f"  {line}",
                    f"  return {text_before_await}",
                    f"\n{text_before_await} = asyncio.run({
                        function_name}())",
                    f"logger.success(format_json({
                        text_before_await}))",
                ])

                updated_lines.append(async_wrapped_code)

            line_idx += 1
        else:
            updated_lines.append(line)
            line_idx += 1

    return "\n".join(updated_lines)


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
            include.lower() in file.lower() for include in include_files)]

    if exclude_files:
        files = [file for file in files if not any(
            exclude.lower() in file.lower() for exclude in exclude_files)]

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
                subfolders = os.path.dirname(file).replace(input_base_dir, '')
                joined_dir = os.path.join(
                    output_dir, subfolders.strip('/'))
                os.makedirs(joined_dir, exist_ok=True)

                output_code_path = os.path.join(joined_dir, f"{file_name}.py")
                for source_group in source_groups:
                    if source_group['type'] == 'text':
                        source_group['code'] = f'"""\n{
                            source_group['code']}\n"""'

                    source_group['code'] = wrap_await_code_multiline_args(
                        source_group['code'])
                    # source_group['code'] = wrap_await_code_singleline_args(
                    #     source_group['code'])

                source_code = "\n\n".join(group['code']
                                          for group in source_groups)

                if with_ollama:
                    source_code = update_code_with_ollama(source_code)
                    source_code = add_general_initializer_code(source_code)
                    source_code = add_ollama_initializer_code(source_code)
                    source_code = add_jet_logger(source_code)
                    source_code = move_all_imports_on_top(source_code)
                    source_code = replace_print_with_jet_logger(source_code)

                if "format_json(" in source_code:
                    source_code = "from jet.transformers.formatters import format_json\n" + source_code

                if "asyncio.run" in source_code:
                    source_code = "import asyncio\n" + source_code

                with open(output_code_path, "w") as f:
                    f.write(source_code)

                logger.log(
                    "Saved code:",
                    output_code_path,
                    colors=["GRAY", "BRIGHT_DEBUG"],
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

    # Sort repo_base_dir by length in descending order
    repo_base_dir = sorted(repo_base_dir, key=len, reverse=True)
    repo_dirs = sorted(repo_dirs, key=len, reverse=True)

    for base_dir in repo_base_dir:
        for repo_dir in repo_dirs:
            repo_path = os.path.join(base_dir, repo_dir)
            if input_base_dir.startswith(repo_path):
                return repo_dir
    return None


def collect_files_and_dirs(input_base_dirs: list[str]) -> (list[str], list[str]):
    include_files = []
    unique_dirs = set()

    for path in input_base_dirs:
        if os.path.isfile(path):
            include_files.append(os.path.basename(path))
            unique_dirs.add(os.path.dirname(path))

    # Convert the set of directories to a sorted list for consistent output
    unique_dirs_list = sorted(unique_dirs)

    return include_files, unique_dirs_list


if __name__ == "__main__":
    repo_base_dir = [
        "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs",
        "/Users/jethroestrada/Desktop/External_Projects/AI/chatbot",
        "/Users/jethroestrada/Desktop/External_Projects/AI/code_agents",
        "/Users/jethroestrada/Desktop/External_Projects/AI/eval_agents",
    ]
    repo_dirs = list_folders(repo_base_dir)
    input_base_dirs = [
        "/Users/jethroestrada/Desktop/External_Projects/AI/code_agents/GenAI_Agents/all_agents_tutorials/research_team_autogen.ipynb",
    ]

    include_files = [
        # "memgraph.ipynb",
    ]
    exclude_files = [
        # "migrating_memory/",
    ]

    collected_results = collect_files_and_dirs(input_base_dirs)
    include_files.extend(collected_results[0])

    input_base_dirs = collected_results[1]

    print("Included Files:", include_files)
    print("Unique Containing Directories:", input_base_dirs)

    extension_mappings = [
        {"ext": [".ipynb"], "output_base_dir": "converted-notebooks"},
        {"ext": [".md", ".mdx"], "output_base_dir": "converted-markdowns"},
    ]

    output_base_dir = os.path.dirname(__file__)

    for input_base_dir in input_base_dirs:
        logger.newline()
        logger.info(f"Processing: {input_base_dir}")

        matching_repo_dir = find_matching_repo_dir(
            input_base_dir, repo_base_dir, repo_dirs)
        logger.log("matching_repo_dir:", matching_repo_dir, colors=[
                   "GRAY", "INFO"])  # Output: "repo1" if found, else None

        if not matching_repo_dir:
            logger.error(f"No matching repo dir: \"{matching_repo_dir}\"")
            continue

        for ext_mapping in extension_mappings:
            extensions = ext_mapping["ext"]
            output_dir = os.path.join(
                output_base_dir,
                matching_repo_dir,
                # ext_mapping["output_base_dir"],
                os.path.basename(input_base_dir),
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
                    "Saved",
                    f"({len(files)})",
                    "files to",
                    output_dir,
                    colors=["WHITE", "SUCCESS", "WHITE", "BRIGHT_SUCCESS"],
                )

            # Remove all triple double quoted comments
            # for item in files:
            #     if item["code_file"].endswith(".py"):
            #         remove_comments(item["code_file"])
