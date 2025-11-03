import os
import shutil
from pathlib import Path
from jet.file.utils import save_file
from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from playwright.sync_api import sync_playwright

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

JS_UTILS_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/scrapers/browser/scripts/utils.js"

def example_page_evaluate(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE)
        page = browser.new_page()
        page.goto(url)

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

        save_file({
            "bounding_box": bounding_box
        }, f"{OUTPUT_DIR}/page_evaluation.json")

def example_inject_js(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE)
        page = browser.new_page()

        # 🔹 Inject listener tracker BEFORE navigation (runs before any page JS)
        page.add_init_script("""
        (() => {
          window.__clickableElements = new Set();
          const origAddEventListener = EventTarget.prototype.addEventListener;
          EventTarget.prototype.addEventListener = function(type, listener, options) {
            if (type === 'click' && this instanceof Element) {
              window.__clickableElements.add(this);
            }
            return origAddEventListener.call(this, type, listener, options);
          };
        })();
        """)

        # Now navigate normally
        page.goto(url)

        # Inject our JS utilities (after load)
        page.add_script_tag(path=JS_UTILS_PATH)

        print("✅ Injected utils.js and click-tracker")

        # ---- Demonstrate all utils ----
        message = page.evaluate("Utils.myInjectedFunction('Jet')")
        print("myInjectedFunction:", message)

        bbox = page.evaluate("Utils.getBoundingBox('h1')")
        print("getBoundingBox('h1'):", bbox)

        scrolled = page.evaluate("Utils.scrollIntoView('h1')")
        print("scrollIntoView('h1'):", scrolled)

        leaf_texts = page.evaluate("Utils.getLeafTexts('body')")
        print("getLeafTexts('body'):", leaf_texts[:5], "..." if len(leaf_texts) > 5 else "")

        clickables = page.evaluate("Utils.getClickableElements()")
        print(f"getClickableElements(): Found {len(clickables)} elements")
        for c in clickables[:3]:
            print(" -", c)

        # ---- Collect elements that had JS click listeners attached ----
        js_clickables = page.evaluate("""
        Array.from(window.__clickableElements).map(el => ({
            tag: el.tagName.toLowerCase(),
            text: el.innerText?.trim().slice(0, 100) || '',
            hasHref: !!el.getAttribute('href')
        }))
        """)
        print(f"Detected {len(js_clickables)} elements with JS click listeners")
        for el in js_clickables[:3]:
            print(" -", el)

        # ---- Capture screenshot for reference ----
        element = page.query_selector("h1")
        if element:
            screenshot_path = Path(os.path.join(OUTPUT_DIR, "injected_h1_screenshot.png")).resolve()
            element.screenshot(path=str(screenshot_path))
            print(f"✅ Screenshot saved at {screenshot_path}")
        else:
            print("❌ Element for screenshot not found")

        browser.close()

def example_inner_element_screenshot(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE)
        page = browser.new_page()
        page.goto(url)

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
    url = "https://gamerant.com/new-isekai-anime-2025"
    example_page_evaluate(url)
    example_inject_js(url)
    example_inner_element_screenshot(url)
