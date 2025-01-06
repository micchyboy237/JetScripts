import os
import fnmatch
import argparse
import subprocess
import json
from _copy_file_structure import (
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
]
include_files = [
    # Client
    # "/Users/jethroestrada/Desktop/External_Projects/PortfolioBaseTemplates/firebase-resume/src/hooks/vector/useQueryNodes.js",
    # Server
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/server/routes/rag.py",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/server/helpers/rag.py",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/server/app.py",
]
structure_include = [
    # "*.py"
]
structure_exclude = []

include_content = []
exclude_content = []

# Args defaults
SHORTEN_FUNCTS = False
INCLUDE_FILE_STRUCTURE = True

DEFAULT_SYSTEM_MESSAGE = """
Dont use or add to memory.
Execute browse or internet search if requested.
""".strip()

DEFAULT_QUERY_MESSAGE = """
Create type for nodes endpoints response given this result:
[
    {
        "score": 0.7249999007515096,
        "node_id": "d5c68ecf-9935-41ac-af20-1b15c2fc5322",
        "metadata": {
            "file_path": "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/summaries/personal_information.md",
            "file_name": "personal_information.md",
            "file_type": "text/markdown",
            "file_size": 573,
            "creation_date": "2025-01-07",
            "last_modified_date": "2025-01-07"
        },
        "text": "The document contains personal information about an individual named Jethro Reuel A. Estrada. This includes his full name, preferred names (Jethro or Jet), nationality (Filipino), gender (Male), age (34 as of 2023), country of residence (Philippines), languages spoken (English and Tagalog), educational background (BS Degree in Computer Engineering from De La Salle University - Manila, graduated in 2012), and contact details such as email address, WhatsApp link, and phone number. The individual's professional position is listed as a Full Stack Web / Mobile Developer.",
        "text_length": 572,
        "start_end": [
            0,
            572
        ],
        "similarity": 0.7249999007515096
    }
]
""".strip()

# Project specific
# DEFAULT_QUERY_MESSAGE += (
#     "\n- Use standard but beautiful designs if html will be provided."
# )

DEFAULT_INSTRUCTIONS_MESSAGE = """
- Provide a step by step process of how you would solve the query.
- Keep the code short, reusable, testable, maintainable and optimized. Follow best practices and industry design patterns.
- Install any libraries required to run the code.
- You may update the code structure if necessary.
- Reuse existing code if possible without breaking anything.
- Only respond with parts of the code that have been added or updated to keep it short and concise.
- Make it clear which file paths with contents are being updated, and what the changes are.
- Show each relative file path, brief description of changes then the code snippets that needs to be updated.
- Include real world usage examples if applicable.
- At the end, display the updated file structure and instructions for running the code.
- Ignore instructions that are not applicable to the query.
""".strip()

# For existing projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
#     "\n- Only respond with parts of the code that have been added or updated to keep it short and concise."
# )

# For creating projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
#     "\n- At the end, display the updated file structure and instructions for running the code."
#     "\n- Provide complete working code for each file (should match file structure)"
# )

# base_dir should be actual file directory
file_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(file_dir)


def find_files(base_dir, include, exclude, include_content_patterns, exclude_content_patterns, case_sensitive=False):
    print("Base Dir:", file_dir)
    print("Finding files:", base_dir, include, exclude)
    include_abs = [
        os.path.relpath(path=pat, start=file_dir)
        if not os.path.isabs(pat) else pat
        for pat in include
        if os.path.exists(os.path.abspath(pat) if not os.path.isabs(pat) else pat)
    ]

    matched_files = set(include_abs)
    for root, dirs, files in os.walk(base_dir):
        # Adjust include and exclude lists: if no wildcard, treat it as a specific file in the current directory
        adjusted_include = [
            os.path.relpath(os.path.join(base_dir, pat), base_dir) if not any(
                c in pat for c in "*?") else pat
            for pat in include
        ]
        adjusted_exclude = [
            os.path.relpath(os.path.join(base_dir, pat), base_dir) if not any(
                c in pat for c in "*?") else pat
            for pat in exclude
        ]

        # Exclude specified directories with or without wildcard support
        dirs[:] = [d for d in dirs if not any(
            fnmatch.fnmatch(d, pat) or fnmatch.fnmatch(os.path.join(root, d), pat) for pat in adjusted_exclude)]

        # Check for files in the current directory that match the include patterns without wildcard support
        for file in files:
            file_path = os.path.relpath(os.path.join(root, file), base_dir)
            if file_path in adjusted_include and not any(fnmatch.fnmatch(file_path, pat) for pat in adjusted_exclude):
                if file_path not in matched_files:
                    matched_files.add(file_path)  # Add to the set
                    print(f"Matched file in current directory: {file_path}")

        # Check for directories that match the include patterns
        for dir_name in dirs:
            dir_path = os.path.relpath(os.path.join(root, dir_name), base_dir)
            if any(fnmatch.fnmatch(dir_name, pat) for pat in adjusted_include) or any(fnmatch.fnmatch(dir_path, pat) for pat in adjusted_include):
                # If the directory matches, find all files within this directory
                for sub_root, _, sub_files in os.walk(os.path.join(root, dir_name).replace("*", "")):
                    # Check if sub_root is excluded
                    base_sub_root = os.path.basename(sub_root)
                    if any(fnmatch.fnmatch(base_sub_root, pat) for pat in adjusted_exclude):
                        break
                    for file in sub_files:
                        file_path = os.path.relpath(
                            os.path.join(sub_root, file), base_dir)
                        if not any(fnmatch.fnmatch(file_path, pat) for pat in adjusted_exclude):
                            if file_path not in matched_files:
                                matched_files.add(file_path)  # Add to the set
                                print(
                                    f"Matched file in directory: {file_path}")

        # Check for files that match the include patterns
        for file in files:
            file_path = os.path.relpath(os.path.join(root, file), base_dir)
            is_current_package_json = (
                file_path == "package.json" and "./package.json" in adjusted_include and root == base_dir)
            if (is_current_package_json or any(fnmatch.fnmatch(file_path, pat) for pat in adjusted_include)) and not any(fnmatch.fnmatch(file_path, pat) for pat in adjusted_exclude):
                # Check if file is excluded
                if file in adjusted_exclude:
                    continue
                # Check file contents against include_content and exclude_content patterns
                full_path = os.path.join(root, file)
                if matches_content(full_path, include_content_patterns, exclude_content_patterns, case_sensitive):
                    if file_path not in matched_files:
                        matched_files.add(file_path)  # Add to the set
                        print(f"Matched file: {file_path}")

        # Check for files in absolute directories that match the include patterns with wildcards
        include_dir_abs = [
            pat for pat in include if "*" in pat and os.path.isdir(pat.replace("*", ""))
        ]

        for dir_name in include_dir_abs:
            # Remove wildcard and calculate relative directory path
            dir_no_wildcard = dir_name.replace("*", "")
            dir_path = os.path.relpath(os.path.join(root, dir_name), base_dir)

            # Check if the directory matches any adjusted include patterns
            if any(fnmatch.fnmatch(dir_name, pat) for pat in adjusted_include) or any(fnmatch.fnmatch(dir_path, pat) for pat in adjusted_include):
                # If matched, find all files within this directory
                for file in os.listdir(os.path.join(root, dir_no_wildcard)):
                    base_file = os.path.basename(file)

                    # Skip if file matches exclude patterns
                    if any(fnmatch.fnmatch(base_file, pat) for pat in adjusted_exclude):
                        continue

                    # Calculate the relative file path and skip if excluded
                    file_path = os.path.relpath(
                        os.path.join(dir_no_wildcard, file), base_dir)
                    if not any(fnmatch.fnmatch(file_path, pat) for pat in adjusted_exclude):
                        # If file path is new, add it to the set
                        rel_file_path = os.path.relpath(file_path)
                        if rel_file_path not in matched_files and os.path.isfile(rel_file_path):
                            matched_files.add(rel_file_path)
                            print(f"Matched file in directory: {
                                  rel_file_path}")

    # Convert the set back to a list before returning
    return list(matched_files)


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
                f"\n// {cleaned_rel_path}\n" if not filenames_only else f"{file}\n")
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
        clipboard_content_parts.append(f"SYSTEM\n{system_message}")
    if instructions_message:
        clipboard_content_parts.append(f"INSTRUCTIONS\n{instructions_message}")
    clipboard_content_parts.append(f"QUERY\n{query_message}")
    if INCLUDE_FILE_STRUCTURE:
        clipboard_content_parts.append(f"FILES STRUCTURE\n{files_structure}")

    if clipboard_content:
        clipboard_content_parts.append(
            f"EXISTING FILES CONTENTS\n{clipboard_content}")

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
