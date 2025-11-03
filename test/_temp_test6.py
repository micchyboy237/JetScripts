import os
import shutil
from pathlib import Path
from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from playwright.sync_api import sync_playwright

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def example_page_evaluate():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE)
        page = browser.new_page()
        page.goto("https://example.com")

        # Evaluate JavaScript in the browser context
        bounding_box = page.evaluate("""
        () => {
            const el = document.querySelector('h1');  // target element
            if (!el) return null;
            const rect = el.getBoundingClientRect();
            return {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
                top: rect.top,
                left: rect.left,
                bottom: rect.bottom,
                right: rect.right
            };
        }
        """)

        print("Bounding box:", bounding_box)

        browser.close()

def example_inject_js():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE)
        page = browser.new_page()
        page.goto("https://example.com")

        current_dir = os.path.dirname(__file__)
        js_path = Path(os.path.join(current_dir, "scripts/utils.js")).resolve()
        page.add_script_tag(path=str(js_path))

        bbox = page.evaluate("Utils.getBoundingBox('h1')")
        print("From injected bounding box:", bbox)

        message = page.evaluate("Utils.myInjectedFunction('Jet')")
        print(message)

        browser.close()

def example_inner_element_screenshot():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE)
        page = browser.new_page()
        page.goto("https://example.com")

        # Locate the element you want to capture
        element = page.query_selector("h1")

        if element:
            screenshot_path = Path(os.path.join(OUTPUT_DIR, "element_screenshot.png")).resolve()
            # Take a screenshot of just that element
            element.screenshot(path=str(screenshot_path))
            print(f"✅ Screenshot saved as {str(screenshot_path)}")
        else:
            print("❌ Element not found")

        browser.close()

if __name__ == "__main__":
    example_page_evaluate()
    example_inject_js()
    example_inner_element_screenshot()
