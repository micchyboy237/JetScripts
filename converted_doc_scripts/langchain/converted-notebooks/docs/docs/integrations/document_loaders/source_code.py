from jet.logger import logger
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import (
Language,
RecursiveCharacterTextSplitter,
)
from langchain_text_splitters import Language
from pprint import pprint
import os
import shutil
import warnings


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Source Code

This notebook covers how to load source code files using a special approach with language parsing: each top-level function and class in the code is loaded into separate documents. Any remaining code top-level code outside the already loaded functions and classes will be loaded into a separate document.

This approach can potentially improve the accuracy of QA models over source code.

The supported languages for code parsing are:

- C (*)
- C++ (*)
- C# (*)
- COBOL
- Elixir
- Go (*)
- Java (*)
- JavaScript (requires package `esprima`)
- Kotlin (*)
- Lua (*)
- Perl (*)
- Python
- Ruby (*)
- Rust (*)
- Scala (*)
- TypeScript (*)

Items marked with (*) require the packages `tree_sitter` and `tree_sitter_languages`.
It is straightforward to add support for additional languages using `tree_sitter`,
although this currently requires modifying LangChain.

The language used for parsing can be configured, along with the minimum number of
lines required to activate the splitting based on syntax.

If a language is not explicitly specified, `LanguageParser` will infer one from
filename extensions, if present.
"""
logger.info("# Source Code")

# %pip install -qU esprima esprima tree_sitter tree_sitter_languages


warnings.filterwarnings("ignore")


loader = GenericLoader.from_filesystem(
    "./example_data/source_code",
    glob="*",
    suffixes=[".py", ".js"],
    parser=LanguageParser(),
)
docs = loader.load()

len(docs)

for document in docs:
    plogger.debug(document.metadata)

logger.debug("\n\n--8<--\n\n".join([document.page_content for document in docs]))

"""
The parser can be disabled for small files. 

The parameter `parser_threshold` indicates the minimum number of lines that the source code file must have to be segmented using the parser.
"""
logger.info("The parser can be disabled for small files.")

loader = GenericLoader.from_filesystem(
    "./example_data/source_code",
    glob="*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=1000),
)
docs = loader.load()

len(docs)

logger.debug(docs[0].page_content)

"""
## Splitting

Additional splitting could be needed for those functions, classes, or scripts that are too big.
"""
logger.info("## Splitting")

loader = GenericLoader.from_filesystem(
    "./example_data/source_code",
    glob="*",
    suffixes=[".js"],
    parser=LanguageParser(language=Language.JS),
)
docs = loader.load()


js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=60, chunk_overlap=0
)

result = js_splitter.split_documents(docs)

len(result)

logger.debug("\n\n--8<--\n\n".join([document.page_content for document in result]))

"""
## Adding Languages using Tree-sitter Template

Expanding language support using the Tree-Sitter template involves a few essential steps:

1. **Creating a New Language File**:
    - Begin by creating a new file in the designated directory (langchain/libs/community/langchain_community/document_loaders/parsers/language).
    - Model this file based on the structure and parsing logic of existing language files like **`cpp.py`**.
    - You will also need to create a file in the langchain directory (langchain/libs/langchain/langchain/document_loaders/parsers/language).
2. **Parsing Language Specifics**:
    - Mimic the structure used in the **`cpp.py`** file, adapting it to suit the language you are incorporating.
    - The primary alteration involves adjusting the chunk query array to suit the syntax and structure of the language you are parsing.
3. **Testing the Language Parser**:
    - For thorough validation, generate a test file specific to the new language. Create **`test_language.py`** in the designated directory(langchain/libs/community/tests/unit_tests/document_loaders/parsers/language).
    - Follow the example set by **`test_cpp.py`** to establish fundamental tests for the parsed elements in the new language.
4. **Integration into the Parser and Text Splitter**:
    - Incorporate your new language within the **`language_parser.py`** file. Ensure to update LANGUAGE_EXTENSIONS and LANGUAGE_SEGMENTERS along with the docstring for LanguageParser to recognize and handle the added language.
    - Also, confirm that your language is included in **`text_splitter.py`** in class Language for proper parsing.

By following these steps and ensuring comprehensive testing and integration, you'll successfully extend language support using the Tree-Sitter template.

Best of luck!
"""
logger.info("## Adding Languages using Tree-sitter Template")

logger.info("\n\n[DONE]", bright=True)