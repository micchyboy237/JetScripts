from jet.code.extraction.extract_definitions import extract_class_and_function_defs
from jet.utils.commands import copy_to_clipboard
from jet.logger import logger
from jet.transformers.formatters import format_json


if __name__ == "__main__":
    result = extract_class_and_function_defs(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques")

    code = ""
    for filepath, defs in result.items():
        code += f"# {filepath}\n"
        for item in defs:
            code += item + "\n"
        code += "\n"  # Add an extra newline after each file block

    logger.gray("Result:")
    logger.success(format_json(result))
    copy_to_clipboard(code)
