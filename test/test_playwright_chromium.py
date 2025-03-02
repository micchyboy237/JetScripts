import re
import unittest
from playwright.sync_api import sync_playwright, Page, expect


class PlaywrightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()
        cls.context = cls.browser.new_context()
        cls.page = cls.context.new_page()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def test_has_title(self):
        self.page.goto("https://playwright.dev/")
        expect(self.page).to_have_title(re.compile("Playwright"))

    def test_get_started_link(self):
        self.page.goto("https://playwright.dev/")
        self.page.get_by_role("link", name="Get started").click()
        expect(self.page.get_by_role(
            "heading", name="Installation")).to_be_visible()


if __name__ == "__main__":
    unittest.main()
