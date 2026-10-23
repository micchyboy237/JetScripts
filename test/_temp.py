import argparse

from trafilatura import extract

if __name__ == "__main__":
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

    with open(html_file, "r", encoding="utf-8") as f:
        html_string = f.read()

    # result = extract(html_string)
    # result = extract(
    #     html_string,
    #     output_format="xml",
    #     include_links=True,
    # )
    # result = extract(
    #     html_string,
    #     output_format="json",
    #     with_metadata=True,
    #     include_links=True,
    # )
    result = extract(
        html_string,
        output_format="markdown",
        include_links=True,
        include_formatting=True,
        url="https://ph.indeed.com",
        # favor_precision=True,
    )
    print(result)
