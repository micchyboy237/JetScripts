import os
import shutil
import asyncio
from jet.scrapers.automation.clone_with_playwright import clone_after_render


async def main():
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    # url = "http://example.com"
    url = "https://aniwatchtv.to"

    await clone_after_render(url, output_dir)
    print('Done with Playwright clone')


if __name__ == '__main__':
    asyncio.run(main())
