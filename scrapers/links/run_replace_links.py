import os
import shutil

from jet.file.utils import load_file, save_file
from jet.scrapers.utils import replace_links

# Example usage
if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0],
    )
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    html_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.html"
    html_str: str = load_file(html_path)

    # Run and save results with base URL
    base_url = "https://deepeval.com"
    result = replace_links(html_str, base=base_url)
    save_file(result, os.path.join(output_dir, "result.html"))
