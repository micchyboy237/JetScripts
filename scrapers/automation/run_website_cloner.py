import os
import shutil

from jet.scrapers.automation.website_cloner import WebsiteCloner
from jet.file.utils import save_file

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    # Example usage
    cloner = WebsiteCloner("https://example.com")
    content = cloner.fetch_website()
    html_output = cloner.generate_tailwind_html()
    print(html_output)

    save_file(html_output, os.path.join(output_dir, 'index.html'))
