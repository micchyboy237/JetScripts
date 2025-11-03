from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from playwright.sync_api import sync_playwright, Page, JSHandle
from typing import List, Dict, Any

def _get_click_listeners_count(page: Page, el: JSHandle) -> int:
    """Helper to get click listener count for a single element."""
    try:
        result = page.evaluate(
            """el => {
                const listeners = getEventListeners(el);
                return listeners.click ? listeners.click.length : 0;
            }""",
            el
        )
        return int(result)
    except Exception:
        return 0

def get_clickable_elements(page: Page) -> List[Dict[str, Any]]:
    """
    Retrieves all elements with click event listeners on the given page.

    Args:
        page: Playwright Page object.

    Returns:
        List of dictionaries with element details.
    """
    try:
        # Get array of all elements as JSHandle
        elements_handle: JSHandle = page.evaluate_handle("() => Array.from(document.querySelectorAll('*'))")
        count: int = page.evaluate("arr => arr.length", elements_handle)
        
        clickable_elements: List[Dict[str, Any]] = []
        
        for i in range(count):
            el: JSHandle = page.evaluate_handle("(arr, i) => arr[i]", [elements_handle, i])
            listeners_count = _get_click_listeners_count(page, el)
            
            if listeners_count > 0:
                props = page.evaluate("""el => ({
                    tagName: el.tagName,
                    id: el.id || '',
                    classes: el.className || '',
                    text: el.textContent ? el.textContent.trim().slice(0, 50) : ''
                })""", el)
                clickable_elements.append(props)
            
            el.dispose()  # Clean up individual element handle
        
        elements_handle.dispose()
        return clickable_elements
    except Exception as e:
        print(f"Error executing script: {e}")
        return []

def main(url: str) -> None:
    """
    Main function to navigate to a URL and print clickable elements.

    Args:
        url: The URL to navigate to.
    """
    with sync_playwright() as p:
        # Use Chromium for getEventListeners support
        browser = p.chromium.launch(
            headless=False,
            executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
        )
        page = browser.new_page()
        
        try:
            page.goto(url, wait_until="domcontentloaded")
            clickable_elements = get_clickable_elements(page)
            
            if clickable_elements:
                print(f"Found {len(clickable_elements)} clickable elements:")
                for i, elem in enumerate(clickable_elements):
                    print(f"{i + 1}. Tag: {elem['tagName']}, ID: {elem['id']}, "
                          f"Classes: {elem['classes']}, Text: {elem['text']}")
            else:
                print("No clickable elements found or an error occurred.")
                
        except Exception as e:
            print(f"Error navigating to {url}: {e}")
        finally:
            browser.close()
            
if __name__ == "__main__":
    # Example usage
    main("https://gamerant.com/new-isekai-anime-2025")