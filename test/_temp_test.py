# import the necessary functions
import os
import shutil
from trafilatura import fetch_url, extract

from jet.file.utils import save_file

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    # grab a HTML file to extract data from
    downloaded = fetch_url(
        'https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/')

    # output main content and comments as plain text
    result = extract(downloaded)
    print(result)

    save_file(result, f"{output_dir}/result.txt")
