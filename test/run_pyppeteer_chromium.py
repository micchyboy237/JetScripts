import asyncio
import os
from jet.logger.timer import sleep_countdown
from pyppeteer import launch


EXECUTABLE_PATH = "/Users/jethroestrada/Library/Caches/ms-playwright/chromium-1140/chrome-mac/Chromium.app/Contents/MacOS/Chromium"
GENERATED_DIR = "generated"
os.makedirs(GENERATED_DIR, exist_ok=True)


async def main():
    browser = await launch(executablePath=EXECUTABLE_PATH, headless=False)
    page = await browser.newPage()
    await page.goto('https://example.com')
    await page.screenshot({'path': f'{GENERATED_DIR}/example.png'})

    dimensions = await page.evaluate('''() => {
        return {
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            deviceScaleFactor: window.devicePixelRatio,
        }
    }''')

    print(dimensions)
    # >>> {'width': 800, 'height': 600, 'deviceScaleFactor': 1}
    sleep_countdown(5)
    await browser.close()

asyncio.get_event_loop().run_until_complete(main())
