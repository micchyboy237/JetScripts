import json

from trafilatura import extract


def extract_with_links_for_rag(html_string: str, url: str = None) -> dict:
    """
    Extract text with embedded links for RAG context
    Links provide additional reference points for the LLM
    """

    # Option 1: Markdown format (best for LLMs - preserves link text and URLs)
    result_md = extract(
        html_string,
        output_format="markdown",
        include_links=True,
        include_formatting=True,
        url=url,
        # favor_precision=True,
    )

    return {"content": result_md, "format": "markdown", "source_url": url}


def extract_structured_links(html_string: str, url: str = None) -> dict:
    """
    Extract text and separately collect all links for RAG
    Useful when you want to track references separately
    """
    from lxml import html as lxml_html

    # Parse HTML to extract links manually
    tree = lxml_html.fromstring(html_string)
    links = []

    for link in tree.xpath("//a[@href]"):
        href = link.get("href")
        text = link.text_content().strip()

        # Convert relative to absolute if base URL provided
        if url and href and not href.startswith(("http://", "https://")):
            from urllib.parse import urljoin

            href = urljoin(url, href)

        if href and text:
            links.append({"url": href, "text": text})

    # Extract main content without links
    content = extract(html_string, url=url, include_links=False, favor_precision=True)

    return {
        "content": content,
        "links": links,
        "link_count": len(links),
        "source_url": url,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract clean text from an HTML file using Trafilatura."
    )
    parser.add_argument(
        "html_file",
        nargs="?",
        default="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/node_extraction/sample.html",
        help="Path to an HTML file to process",
    )
    args = parser.parse_args()
    html_file = args.html_file

    url = "https://ph.indeed.com"

    with open(html_file, "r", encoding="utf-8") as f:
        html_string = f.read()

    print("Method 1: Markdown with inline links")
    rag_context = extract_with_links_for_rag(html_string, url=url)
    print(rag_context["content"])

    print("\n\n")

    print("Method 2: Structured links + clean text")
    structured = extract_structured_links(html_string, url=url)
    print(json.dumps(structured, indent=2))
