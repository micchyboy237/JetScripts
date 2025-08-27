import os
import shutil
from jet.code.doc_parsers.rst_parser import parse_rst_to_blocks
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    # quick demo
    sample1 = """
Title
=====

Paragraph text.

- bullet 1
- bullet 2

.. note::

   This is an admonition.

::

    a literal block

.. rubric:: A rubric

.. parsed-literal::

   parsed literal text
"""
    parsed_blocks1 = parse_rst_to_blocks(sample1)
    logger.success(format_json(parsed_blocks1))
    save_file(sample1, f"{OUTPUT_DIR}/sample1.rst")
    save_file(parsed_blocks1, f"{OUTPUT_DIR}/parsed_blocks1.json")

    rst_file = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/TextBlob/docs/advanced_usage.rst"
    sample2 = load_file(rst_file)
    parsed_blocks2 = parse_rst_to_blocks(sample2)
    logger.success(format_json(parsed_blocks2))
    save_file(sample2, f"{OUTPUT_DIR}/sample2.rst")
    save_file(parsed_blocks2, f"{OUTPUT_DIR}/parsed_blocks2.json")

    sample3 = """
Tokenizers
----------

New in version `0.4.0`.

The ``words`` and ``sentences`` properties are helpers that use the ``textblob.tokenizers.WordTokenizer`` and ``textblob.tokenizers.SentenceTokenizer`` classes, respectively.

You can use other tokenizers, such as those provided by NLTK, by passing them into the ``TextBlob`` constructor then accessing the ``tokens`` property.

.. doctest::

    >>> from textblob import TextBlob
    >>> from nltk.tokenize import TabTokenizer
    >>> tokenizer = TabTokenizer()
    >>> blob = TextBlob("This is\ta rather tabby\tblob.", tokenizer=tokenizer)
    >>> blob.tokens
    WordList(['This is', 'a rather tabby', 'blob.'])    
"""
    parsed_blocks3 = parse_rst_to_blocks(sample3)
    logger.success(format_json(parsed_blocks3))
    save_file(sample3, f"{OUTPUT_DIR}/sample3.rst")
    save_file(parsed_blocks3, f"{OUTPUT_DIR}/parsed_blocks3.json")
