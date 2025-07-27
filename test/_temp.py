import shutil
from typing import List

from jet.code.markdown_types.markdown_parsed_types import MarkdownToken
from jet.code.markdown_utils import parse_markdown
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy, prepend_missing_headers_by_type
from jet.file.utils import load_file, save_file
from jet.utils.print_utils import print_dict_types

import os
OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

md_tokens: List[MarkdownToken] = [
    {
        "content": "Personal Information",
        "level": 2,
        "line": 1,
        "type": "header",
        "meta": {}
    },
    {
        "content": "Contact Details",
        "line": 2,
        "type": "header",
        "level": 2,
        "meta": {}
    },
    {
        "line": 3,
        "meta": {
            "items": [
                {
                    "text": "Full Name: Jethro Reuel A. Estrada",
                    "task_item": False
                }
            ]
        },
        "type": "unordered_list",
        "content": "",
        "level": None
    },
    {
        "content": "Personal Details",
        "line": 4,
        "type": "header",
        "level": 2,
        "meta": {}
    },
    {
        "line": 5,
        "meta": {
            "items": [
                {
                    "text": "Gender: Male",
                    "task_item": False
                }
            ]
        },
        "type": "unordered_list",
        "content": "",
        "level": None
    },
]

expected =
if __name__ == "__main__":
    results = prepend_missing_headers_by_type(md_tokens)
    save_file(results, f"{OUTPUT_DIR}/results.json")

    assert results == expected
