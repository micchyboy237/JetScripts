import shutil
from urllib.parse import urlparse, unquote
import re
from typing import List, Dict, Optional

from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.transformers.link_formatters import LinkFormatter
import os
from jet.file.utils import load_file, save_file
from jet.llm.utils.search_docs import search_docs

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    query = "Python is a popular programming language."
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models require large datasets.",
        "Python is a popular programming language.",
        "Neural networks are used in machine learning."
    ]
    ids = ["doc1", "doc2", "doc3", "doc4"]

    search_doc_results = search_docs(
        query=query,
        documents=documents,
        model="all-minilm:33m",
        top_k=None,
        ids=ids
    )

    logger.success(format_json(search_doc_results))

    save_file(
        {"query": query, "count": len(
            search_doc_results), "results": search_doc_results},
        os.path.join(output_dir, "search_doc_results.json")
    )
