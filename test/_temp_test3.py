import ast
import os
from typing import Dict, List, Union


def extract_class_and_function_defs(target_dir: str) -> Dict[str, List[str]]:
    """
    Recursively extracts class and function headers and docstrings from .py files.

    Args:
        target_dir (str): The directory to search.

    Returns:
        Dict[str, List[str]]: A mapping of file paths to a list of stripped defs.
    """
    def_nodes = {}

    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    lines = source.splitlines()
                    tree = ast.parse(source, filename=full_path)

                    extracted = []

                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            # Extract signature line
                            header = lines[node.lineno - 1].rstrip()

                            # Extract docstring if present
                            docstring = ast.get_docstring(node, clean=False)
                            if docstring:
                                # Get the line numbers of the docstring
                                first_stmt = node.body[0]
                                if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Str):
                                    doc_lines = lines[first_stmt.lineno -
                                                      1:first_stmt.end_lineno]
                                else:
                                    doc_lines = []
                            else:
                                doc_lines = []

                            extracted.append("\n".join([header] + doc_lines))

                    if extracted:
                        def_nodes[full_path] = extracted

                except (SyntaxError, UnicodeDecodeError) as e:
                    print(f"Error processing {full_path}: {e}")

    return def_nodes
