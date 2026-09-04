import json
import logging
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.logging import RichHandler

# --- Reused setup ---
console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Rich logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
)
logger = logging.getLogger("chrome_tab_extractor")

CDP_URL = "http://localhost:9222"
ZOOM_LEVEL = 0.75  # 75%


def set_native_zoom(context, zoom_level: float):
    """Sets Chrome's real native zoom (Ctrl+Minus equivalent) profile-wide."""
    logger.info(
        f"Setting native browser zoom to [bold]{int(zoom_level * 100)}%[/bold]..."
    )
    settings_page = context.new_page()
    try:
        settings_page.goto("chrome://settings")
        settings_page.evaluate(f"chrome.settingsPrivate.setDefaultZoom({zoom_level})")
        logger.info("[green]Native zoom applied.[/green]")
    except Exception as e:
        logger.warning(
            f"Native zoom failed ([red]{e}[/red]) — will use CSS zoom fallback per tab."
        )
    finally:
        settings_page.close()


def apply_css_zoom_fallback(page, zoom_level: float):
    """Belt-and-suspenders: also apply CSS zoom in case native zoom didn't stick."""
    try:
        page.evaluate(f"document.body.style.zoom = '{int(zoom_level * 100)}%'")
    except Exception as e:
        logger.warning(f"  CSS zoom fallback failed for this tab: {e}")


def safe_filename(text: str, max_len: int = 50) -> str:
    """Turn a page title into a filesystem-safe basename."""
    cleaned = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in text)
    cleaned = "_".join(cleaned.split())
    return cleaned[:max_len] or "untitled"


def extract_tabs_and_screenshot() -> list[Path]:
    saved_files: list[Path] = []

    with sync_playwright() as p:
        logger.info(f"Connecting to Chrome via CDP at [cyan]{CDP_URL}[/cyan] ...")
        browser = p.chromium.connect_over_cdp(CDP_URL, timeout=30000, is_local=True)
        logger.info("[green]Connected successfully.[/green]")

        context = browser.contexts[0]
        set_native_zoom(context, ZOOM_LEVEL)

        pages = context.pages
        logger.info(f"Found [bold]{len(pages)}[/bold] tab(s) open.")

        tabs_info = []

        for tab_index, page in enumerate(pages):
            try:
                title = page.title()
                url = page.url
                logger.info(
                    f"Tab {tab_index}: [bold]{title}[/bold] -> [dim]{url}[/dim]"
                )

                apply_css_zoom_fallback(page, ZOOM_LEVEL)

                base_name = f"{tab_index:02d}_{safe_filename(title)}"

                # --- Screenshot ---
                screenshot_path = OUTPUT_DIR / f"{base_name}.png"
                logger.info(f"  📸 Capturing screenshot -> {screenshot_path.name}")
                page.screenshot(path=str(screenshot_path), full_page=True)
                saved_files.append(screenshot_path)

                # --- HTML ---
                html_path = OUTPUT_DIR / f"{base_name}.html"
                logger.info(f"  📝 Saving HTML -> {html_path.name}")
                html_path.write_text(page.content(), encoding="utf-8")
                saved_files.append(html_path)

                tabs_info.append(
                    {
                        "tab_index": tab_index,
                        "title": title,
                        "url": url,
                        "screenshot": screenshot_path.name,
                        "html": html_path.name,
                    }
                )

            except Exception as e:
                logger.error(f"  [red]Failed to process tab {tab_index}: {e}[/red]")

        # Save combined metadata as JSON
        info_path = OUTPUT_DIR / "tabs_info.json"
        info_path.write_text(json.dumps(tabs_info, indent=2, ensure_ascii=False))
        saved_files.append(info_path)
        logger.info(f"📄 Saved tab metadata -> {info_path.name}")

    return saved_files


def show_saved_files(saved_files: list[Path]):
    console.print("\n[bold underline]Saved files[/bold underline]")
    for file_path in saved_files:
        uri = file_path.resolve().as_uri()
        console.print(f"  [link={uri}]{file_path.name}[/link]")


if __name__ == "__main__":
    files = extract_tabs_and_screenshot()
    show_saved_files(files)
