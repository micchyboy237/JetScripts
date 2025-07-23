import asyncio
import os
from pathlib import Path
import shutil

from jet.scrapers.automation.webpage_cloner import clone_after_render, generate_react_components


async def main():
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    url = "http://example.com"

    # Clone webpage
    await clone_after_render(url, output_dir, headless=False)

    # Generate React components
    html_path = Path(output_dir) / "index.html"
    html_content = html_path.read_text(encoding="utf-8")
    generate_react_components(html_content, output_dir)

if __name__ == "__main__":
    asyncio.run(main())
