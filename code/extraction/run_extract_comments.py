from jet.code.extraction.extract_comments import extract_comments
from jet.utils.commands import copy_to_clipboard
from jet.logger import logger

if __name__ == "__main__":
    """Main function to process notebook files."""
    notebook_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/notebooks"

    include_outputs = False
    include_code = False
    include_comments = True

    content = extract_comments(
        notebook_path,
    )

    logger.gray("Result:")
    copy_to_clipboard(content)
