from jet.logger import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil


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
---
keywords: [recursivecharactertextsplitter]
---

# How to recursively split text by characters

This [text splitter](/docs/concepts/text_splitters/) is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is `["\n\n", "\n", " ", ""]`. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

1. How the text is split: by list of characters.
2. How the chunk size is measured: by number of characters.

Below we show example usage.

To obtain the string content directly, use `.split_text`.

To create LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects (e.g., for use in downstream tasks), use `.create_documents`.
"""
logger.info("# How to recursively split text by characters")

# %pip install -qU langchain-text-splitters


with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([state_of_the_union])
logger.debug(texts[0])
logger.debug(texts[1])

text_splitter.split_text(state_of_the_union)[:2]

"""
Let's go through the parameters set above for `RecursiveCharacterTextSplitter`:
- `chunk_size`: The maximum size of a chunk, where size is determined by the `length_function`.
- `chunk_overlap`: Target overlap between chunks. Overlapping chunks helps to mitigate loss of information when context is divided between chunks.
- `length_function`: Function determining the chunk size.
- `is_separator_regex`: Whether the separator list (defaulting to `["\n\n", "\n", " ", ""]`) should be interpreted as regex.

## Splitting text from languages without word boundaries

Some writing systems do not have [word boundaries](https://en.wikipedia.org/wiki/Category:Writing_systems_without_word_boundaries), for example Chinese, Japanese, and Thai. Splitting text with the default separator list of `["\n\n", "\n", " ", ""]` can cause words to be split between chunks. To keep words together, you can override the list of separators to include additional punctuation:

* Add ASCII full-stop "`.`", [Unicode fullwidth](https://en.wikipedia.org/wiki/Halfwidth_and_Fullwidth_Forms_(Unicode_block)) full stop "`．`" (used in Chinese text), and [ideographic full stop](https://en.wikipedia.org/wiki/CJK_Symbols_and_Punctuation) "`。`" (used in Japanese and Chinese)
* Add [Zero-width space](https://en.wikipedia.org/wiki/Zero-width_space) used in Thai, Myanmar, Kmer, and Japanese.
* Add ASCII comma "`,`", Unicode fullwidth comma "`，`", and Unicode ideographic comma "`、`"
"""
logger.info("## Splitting text from languages without word boundaries")

text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
)

logger.info("\n\n[DONE]", bright=True)