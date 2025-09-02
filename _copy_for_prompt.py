import os
import fnmatch
import argparse
import subprocess
import json
from _copy_file_structure import (
    find_files,
    format_file_structure,
    clean_newlines,
    clean_content,
    remove_parent_paths
)
from jet.logger import logger

exclude_files = [
    ".git",
    ".gitignore",
    ".DS_Store",
    "_copy*.py",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "*lock.json",
    "*.lock",
    "public",
    "mocks",
    ".venv",
    "dream",
    "jupyter",
    ".*",
    "_*",
    # "generated",
    # Custom
    "*response*",
    "*.sh"
]
include_files = [
    # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/vectors/semantic_search/file_vector_search.py",
    # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/vectors/semantic_search/text_vector_search.py",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/servers/mcp/mcp_server.py",
]
structure_include = [
    # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/tasks/*.py",
    # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data",
]
structure_exclude = []

include_content = []
exclude_content = []

# Args defaults
SHORTEN_FUNCTS = False
INCLUDE_FILE_STRUCTURE = False

DEFAULT_QUERY_MESSAGE = """
Now update mcp_server to add a new tool search_texts using text_vector_search.
""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
""".strip()

DEFAULT_SYSTEM_MESSAGE = """
Dont use prior artifact knowledge and memory.
Only provide updated parts.
""".strip()

# For existing projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
#     "\n- Only respond with parts of the code that have been added or updated to keep it short and concise."
# )z

# For creating projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
#     "\n- At the end, display the updated file structure and instructions for running the code."
#     "\n- Provide complete working code for each file (should match file structure)"
# )

# base_dir should be actual file directory
file_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(file_dir)


def matches_content(file_path, include_patterns, exclude_patterns, case_sensitive=False):
    """
    Check if the file content matches include_patterns and does not match exclude_patterns.
    """
    if not include_patterns and not exclude_patterns:
        return True
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            if not case_sensitive:
                # Convert content to lowercase for case-insensitive matching
                content = content.lower()

            # Check for include content patterns
            if include_patterns:
                include_patterns = [
                    pattern if case_sensitive else pattern.lower() for pattern in include_patterns]
                if not any((fnmatch.fnmatch(content, pattern) if '*' in pattern or '?' in pattern else pattern in content) for pattern in include_patterns):
                    return False

            # Check for exclude content patterns
            if exclude_patterns:
                exclude_patterns = [
                    pattern if case_sensitive else pattern.lower() for pattern in exclude_patterns]
                if any((fnmatch.fnmatch(content, pattern) if '*' in pattern or '?' in pattern else pattern in content) for pattern in exclude_patterns):
                    return False

        return True
    except (OSError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
        return False


def main():
    global exclude_files, include_files, include_content, exclude_content

    print("Running _copy_for_prompt.py")
    # Parse command-line options
    parser = argparse.ArgumentParser(
        description='Generate clipboard content from specified files.')
    parser.add_argument('-b', '--base-dir', default=file_dir,
                        help='Base directory to search files in (default: current directory)')
    parser.add_argument('-if', '--include-files', nargs='*', default=include_files,
                        help='Patterns of files to include (default: schema.prisma, episode)')
    parser.add_argument('-ef', '--exclude-files', nargs='*', default=exclude_files,
                        help='Directories or files to exclude (default: node_modules)')
    parser.add_argument('-ic', '--include-content', nargs='*', default=include_content,
                        help='Patterns of file content to include')
    parser.add_argument('-ec', '--exclude-content', nargs='*', default=exclude_content,
                        help='Patterns of file content to exclude')
    parser.add_argument('-cs', '--case-sensitive', action='store_true', default=False,
                        help='Make content pattern matching case-sensitive')
    parser.add_argument('-sf', '--shorten-funcs', action='store_true', default=SHORTEN_FUNCTS,
                        help='Shorten function and class definitions')
    parser.add_argument('-s', '--system', default=DEFAULT_SYSTEM_MESSAGE,
                        help='Message to include in the clipboard content')
    parser.add_argument('-m', '--message', default=DEFAULT_QUERY_MESSAGE,
                        help='Message to include in the clipboard content')
    parser.add_argument('-i', '--instructions', default=DEFAULT_INSTRUCTIONS_MESSAGE,
                        help='Instructions to include in the clipboard content')
    parser.add_argument('-fo', '--filenames-only', action='store_true',
                        help='Only copy the relative filenames, not their contents')
    parser.add_argument('-nl', '--no-length', action='store_true', default=INCLUDE_FILE_STRUCTURE,
                        help='Do not show file character length')

    args = parser.parse_args()
    base_dir = args.base_dir
    include = args.include_files
    exclude = args.exclude_files
    include_content = args.include_content
    exclude_content = args.exclude_content
    case_sensitive = args.case_sensitive
    shorten_funcs = args.shorten_funcs
    query_message = args.message
    system_message = args.system
    instructions_message = args.instructions
    filenames_only = args.filenames_only
    show_file_length = not args.no_length

    # Find all files matching the patterns in the base directory and its subdirectories
    print("\n")
    context_files = find_files(base_dir, include, exclude,
                               include_content, exclude_content, case_sensitive)

    print("\n")
    print(f"Include patterns: {include}")
    print(f"Exclude patterns: {exclude}")
    print(f"Include content patterns: {include_content}")
    print(f"Exclude content patterns: {exclude_content}")
    print(f"Case sensitive: {case_sensitive}")
    print(f"Filenames only: {filenames_only}")
    print(f"\nFound files ({len(context_files)}):\n{
          json.dumps(context_files, indent=2)}")

    print("\n")

    # Initialize the clipboard content
    clipboard_content = ""

    if not context_files:
        print("No context files found matching the given patterns.")
    else:

        # Append relative filenames to the clipboard content
        for file in context_files:
            rel_path = os.path.relpath(path=file, start=file_dir)
            cleaned_rel_path = remove_parent_paths(rel_path)

            prefix = (
                f"\n# {cleaned_rel_path}\n" if not filenames_only else f"{file}\n")
            if filenames_only:
                clipboard_content += f"{prefix}"
            else:
                file_path = os.path.relpath(os.path.join(base_dir, file))
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            content = clean_content(
                                content, file, shorten_funcs)
                            clipboard_content += f"{prefix}{content}\n\n"
                    except Exception:
                        # Continue to the next file
                        continue
                else:
                    clipboard_content += f"{prefix}\n"

        clipboard_content = clean_newlines(clipboard_content).strip()

    # Generate and format the file structure
    structure_include_files = structure_include
    if include:
        structure_include_files += include
    structure_exclude_files = structure_exclude
    if exclude:
        structure_exclude_files += exclude
    files_structure = format_file_structure(
        base_dir,
        include_files=structure_include_files,
        exclude_files=structure_exclude_files,
        include_content=include_content,
        exclude_content=exclude_content,
        case_sensitive=case_sensitive,
        shorten_funcs=shorten_funcs,
        show_file_length=show_file_length,
    )

    # Prepend system and query to the clipboard content then append instructions
    clipboard_content_parts = []

    if system_message:
        clipboard_content_parts.append(f"System\n{system_message}")
    if instructions_message:
        clipboard_content_parts.append(f"Instructions\n{instructions_message}")
    clipboard_content_parts.append(f"Query\n{query_message}")
    if INCLUDE_FILE_STRUCTURE:
        clipboard_content_parts.append(f"Files Structure\n{files_structure}")

    if clipboard_content:
        clipboard_content_parts.append(
            f"Existing Files Contents\n{clipboard_content}")

    clipboard_content = "\n\n".join(clipboard_content_parts)

    # Copy the content to the clipboard
    process = subprocess.Popen(
        'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(clipboard_content.encode('utf-8'))

    # Print the copied content character count
    logger.log("Prompt Char Count:", len(clipboard_content),
               colors=["GRAY", "SUCCESS"])

    # Newline
    print("\n")


if __name__ == "__main__":
    main()
