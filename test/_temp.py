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

md_content = """\
## 📌 Personal Information

### Contact Details

- **Full Name**: Jethro Reuel A. Estrada
- **Preferred Name**: Jethro / Jet
- **Email**: [jethroestrada237@gmail.com](mailto:jethroestrada237@gmail.com)
- **Phone / WhatsApp**: +63 910 166 2460 ([WhatsApp Link](https://wa.me/639101662460))
- **Location**: Las Piñas, Metro Manila, Philippines, 1740

### Personal Details

- **Gender**: Male
- **Nationality**: Filipino
- **Date of Birth**: December 1, 1990
- **Languages Spoken**: English, Tagalog
""".strip()


if __name__ == "__main__":
    markdown_tokens = tokens = base_parse_markdown(
        md_content, ignore_links=False)
    save_file(markdown_tokens, f"{OUTPUT_DIR}/markdown_tokens.json")

    markdown_tokens_with_new_headers = tokens = prepend_missing_headers_by_type(
        markdown_tokens)
    save_file(markdown_tokens_with_new_headers,
              f"{OUTPUT_DIR}/markdown_tokens_with_new_headers.json")

    header_docs = derive_by_header_hierarchy(md_content)
    save_file(header_docs, f"{OUTPUT_DIR}/header_docs.json")
