import json
import os
import argparse
import subprocess
import re

DEFAULT_FILE = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/langchain/libs/langchain/langchain/chains/conversation/memory.py"
DEFAULT_PYTHON_PATH = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/langchain/libs/langchain/langchain"

DEFAULT_SYSTEM_MESSAGE = """
Don't use memory
""".strip()

DEFAULT_MESSAGE = """
Refactor, optimize and reduce the code into one merged base file. Try to reduce the amount of code. Make the code readable and maintainable. Use the latest features of TypeScript and React. Make sure the code is well-documented and follows best practices.
""".strip()

file_dir = os.path.dirname(os.path.realpath(__file__))

MAX_FILE_CHAR_COUNT = None
MAX_COPIED_CHAR_COUNT = None


def clean_content(content):
    """Removes comments, unused imports, and reduces consecutive newlines."""
    content = clean_comments(content)
    content = clean_imports(content)
    content = clean_newlines(content)
    return content.strip()


def clean_comments(content):
    """Removes comments from the given content."""
    return re.sub(r'//.*?$|/\*.*?\*/', '', content, flags=re.MULTILINE | re.DOTALL)


def clean_newlines(content):
    """Removes consecutive newlines from the given content."""
    return re.sub(r'\n\s*\n+', '\n', content)


def clean_imports(content):
    """Removes import statements from the given content."""
    return re.sub(r'import\s+.*?from\s+["\'].*?["\'];?', '', content, flags=re.MULTILINE)


def find_imports(file_path, base_dir):
    """Finds all import statements in a given Python file and returns a list of imported files with their paths."""
    imported_files = {}
    import_pattern = re.compile(
        r'^(?:from\s+([a-zA-Z0-9_.]+)\s+import)|(?:import\s+([a-zA-Z0-9_.]+))', re.MULTILINE
    )

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Debug: print file content
    print(f"Content of {file_path}:\n{content}\n")

    matches = import_pattern.findall(content)

    # Debug: print matches
    print(f"Matches found: {matches}")

    for match in matches:
        # Extract the module path from the correct group
        module = match[0] or match[1]  # Only two groups in the regex
        if module:
            module_path = module.replace('.', '/')
            imported_file_path = os.path.join(base_dir, module_path) + ".py"

            # If the file exists, add it to the imports
            if os.path.exists(imported_file_path):
                imported_files[module] = imported_file_path
            else:
                # Check if it's a package (directory with __init__.py)
                package_path = os.path.join(base_dir, module_path)
                init_file = os.path.join(package_path, "__init__.py")
                if os.path.exists(init_file):
                    imported_files[module] = init_file

    # Debug: print resolved imports
    print(f"Resolved imports: {imported_files}\n")
    return imported_files


def locate_actual_definitions(file_path, base_dir, max_depth=1, max_char=MAX_FILE_CHAR_COUNT):
    """Recursively finds the actual definitions of imported variables or components in the given file up to a specified depth."""
    read_files = set()

    def recursive_imports(file, current_depth, indent=""):
        if file in read_files or current_depth > max_depth:
            return ""  # Avoid cyclic imports or exceed depth

        read_files.add(file)
        file_type = "Base file" if len(read_files) == 1 else "Imported file"
        relative_file_path = os.path.relpath(file, base_dir)
        content = f"\n\n## {file_type}: {relative_file_path} ##\n"

        with open(file, 'r') as f:
            file_content = f.read()
            if max_char:
                content += file_content[:max_char] + "\n\n[More]\n"
            else:
                content += file_content

        if current_depth < max_depth:
            file_imports = find_imports(file, base_dir)
            for import_name, import_file in file_imports.items():
                if re.search(rf'\b{re.escape(import_name.split("/")[-1])}\b', file_content):
                    content += recursive_imports(import_file,
                                                 current_depth + 1, indent + "  ")

        return content

    merged_content = recursive_imports(file_path, 1)
    content = clean_content(merged_content)
    return {
        "content": content,
        "paths": [
            file_path,
            *list(read_files),
        ],
    }


def main():
    print("Running _copy_combined_code.py")

    parser = argparse.ArgumentParser(
        description='Generate clipboard content from specified files.')
    parser.add_argument('file', nargs='?', default=DEFAULT_FILE,
                        help='File path to process (e.g., index.tsx, App.tsx)')
    parser.add_argument('-b', '--base-dir', default=os.path.dirname(DEFAULT_PYTHON_PATH),
                        help='Base dir for resolving non-relative imports')
    parser.add_argument('-s', '--system', default=DEFAULT_SYSTEM_MESSAGE,
                        help='Message to include in the clipboard content')
    parser.add_argument('-m', '--message', default=DEFAULT_MESSAGE,
                        help='Message to include in the clipboard content')
    parser.add_argument('-d', '--deep', type=int, default=2,
                        help='Maximum recursive depth for imports')

    args = parser.parse_args()

    input_file = os.path.abspath(args.file)
    base_dir = os.path.abspath(
        args.base_dir) if args.base_dir else os.path.dirname(input_file)
    max_depth = args.deep

    clipboard_content = f"{args.system}\n\n{args.message}\n\n"
    result = locate_actual_definitions(
        input_file, base_dir, max_depth)

    clipboard_content += result["content"]
    copied_paths = result["paths"]

    if MAX_COPIED_CHAR_COUNT and len(clipboard_content) > MAX_COPIED_CHAR_COUNT:
        clipboard_content = clipboard_content[:MAX_COPIED_CHAR_COUNT]
        clipboard_content += "\n\n[Content truncated due to length]\n"

    # Print the content to the console for manual copying (if needed)
    print(clipboard_content)
    # Print content count
    print(f"\n\nCopied Paths ({len(copied_paths)}):\n{
          json.dumps(copied_paths, indent=2)}")
    print(f"\n\nPrompt Char Count: {len(clipboard_content)}")

    # Copy the content to the clipboard
    subprocess.run('pbcopy', input=clipboard_content.encode(
        'utf-8'), check=True, env={'LANG': 'en_US.UTF-8'})


if __name__ == "__main__":
    main()


# Sample usage:
# python _copy_combined_code.py src/components/baseInput/index.tsx
