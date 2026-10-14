import asyncio

from playwright.async_api import async_playwright


async def capture_snapshot(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Navigate to a page
        await page.goto(url)

        # ✅ CORRECT: Use aria_snapshot() instead of accessibility.snapshot()
        snapshot = await page.aria_snapshot()

        print(snapshot)

        await browser.close()


url = "https://ph.talent.com"
asyncio.run(capture_snapshot(url))
