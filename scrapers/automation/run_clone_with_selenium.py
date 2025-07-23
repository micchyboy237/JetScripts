import os
import shutil
from jet.scrapers.automation.clone_with_selenium import clone_after_render


if __name__ == '__main__':
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    # url = "http://example.com"
    url = "https://aniwatchtv.to"

    clone_after_render(url, output_dir)
    print('Done with Selenium clone')
