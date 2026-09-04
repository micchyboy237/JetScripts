import logging

from playwright.sync_api import sync_playwright

# ---- Logging setup for traceability ----
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("chrome_tab_reader")

CDP_URL = "http://localhost:9222"


def get_open_tabs():
    """
    Connects to an already-running Chrome instance (started with
    --remote-debugging-port=9222) and returns info on every open
    window (context) and tab (page).
    """
    with sync_playwright() as p:
        logger.info(f"Connecting to Chrome via CDP at {CDP_URL} ...")
        browser = p.chromium.connect_over_cdp(CDP_URL)
        logger.info("Connected successfully.")

        all_tabs = []

        # Each "context" = one Chrome window (or profile)
        contexts = browser.contexts
        logger.info(f"Found {len(contexts)} open window(s)/context(s).")

        for window_index, context in enumerate(contexts):
            pages = context.pages  # all tabs inside this window
            logger.info(f"Window {window_index}: {len(pages)} tab(s) open.")

            for tab_index, page in enumerate(pages):
                try:
                    title = page.title()
                    url = page.url
                    logger.info(f"  Tab {tab_index}: '{title}' -> {url}")

                    all_tabs.append(
                        {
                            "window_index": window_index,
                            "tab_index": tab_index,
                            "title": title,
                            "url": url,
                        }
                    )
                except Exception as e:
                    logger.warning(f"  Could not read tab {tab_index}: {e}")

        # NOTE: we don't call browser.close() here —
        # that would close the user's real Chrome window!
        return all_tabs


def watch_for_new_tabs():
    """
    Optional: keep the script running and log new tabs as they open.
    Press Ctrl+C to stop.
    """
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_URL)
        context = browser.contexts[0]  # first window

        def on_new_page(page):
            logger.info(f"New tab opened: {page.url}")

        context.on("page", on_new_page)

        logger.info("Watching for new tabs... (Ctrl+C to stop)")
        try:
            while True:
                pass  # keep script alive
        except KeyboardInterrupt:
            logger.info("Stopped watching.")


if __name__ == "__main__":
    tabs = get_open_tabs()
    print("\n--- Summary ---")
    for tab in tabs:
        print(tab)
