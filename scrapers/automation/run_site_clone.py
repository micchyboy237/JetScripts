import os
import shutil
from jet.scrapers.automation.site_clone import clone_site


if __name__ == '__main__':
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    # url = "http://example.com"
    url = "https://aniwatchtv.to"

    clone_site(url, f'{output_dir}/mirror')
    print('Done cloning:', url)
