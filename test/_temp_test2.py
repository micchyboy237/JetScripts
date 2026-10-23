from trafilatura import bare_extraction


def extract_with_links_for_rag(html_string: str, url: str = None) -> dict:
    """
    Extract content preserving links for RAG context
    """
    doc = bare_extraction(html_string, url=url, include_comments=False)

    # Get all links from the body
    links = []
    if doc.body is not None:
        for link in doc.body.iter("a"):
            href = link.get("href")
            text = link.text_content().strip()
            if href and text:
                # Convert relative to absolute if needed
                if href.startswith("/") and url:
                    from urllib.parse import urljoin

                    href = urljoin(url, href)
                links.append({"text": text, "url": href})

    return {"title": doc.title, "text": doc.text, "links": links, "source_url": url}


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

    result = extract_with_links_for_rag(html_string, url=url)

    print("1. Text result")
    print(result["text"])
    # Output: "RAG Systems Guide\nLearn about RAG and vector databases."

    print("\n\n")

    print("2. Links result")
    print(result["links"])
    # Output: [{'text': 'RAG', 'url': 'https://example.com/rag'},
    #          {'text': 'vector databases', 'url': 'https://example.com/vector-databases'}]
