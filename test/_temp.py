import os
import re

from typing import AsyncGenerator, List, Optional, Tuple, TypedDict
from pyquery import PyQuery as pq

from playwright.sync_api import sync_playwright


class BaseNode:
    """Base class for nodes with common attributes."""

    def __init__(
        self,
        tag: str,
        text: Optional[str],
        depth: int,
        id: str,
        parent: Optional[str] = None,
        class_names: List[str] = [],
        link: Optional[str] = None,
        line: int = 0
    ):
        self.tag = tag
        self.text = text
        self.depth = depth
        self.id = id
        self.parent = parent
        self.class_names = class_names
        self.link = link
        self.line = line


def extract_text_elements(source: str, excludes: list[str] = ["nav", "footer", "script", "style"], timeout_ms: int = 1000) -> List[str]:
    """
    Extracts a flattened list of text elements from the HTML document, ignoring specific elements like <style> and <script>.
    Uses Playwright to render dynamic content if needed.

    :param source: The HTML string or URL to parse.
    :param excludes: A list of tag names to exclude (e.g., ["nav", "footer", "script", "style"]).
    :param timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
    :return: A list of text elements found in the HTML.
    """
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"

    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source

    # Use Playwright to render the page if URL is provided
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)

        page.wait_for_timeout(timeout_ms)

        # Extract the content
        page_content = page.content()
        browser.close()

    # Parse the content with PyQuery after Playwright has rendered it
    doc = pq(page_content)

    # Apply the exclusion logic before extracting text
    exclude_elements(doc, excludes)

    def extract_text(element) -> List[str]:
        text = pq(element).text().strip()

        valid_id_pattern = r'^[a-zA-Z_-]+$'
        element_id = pq(element).attr('id')
        element_class = pq(element).attr('class')
        id = element_id if element_id and re.match(
            valid_id_pattern, element_id) else None
        class_names = [name for name in (element_class.split() if element_class else [])
                       if re.match(valid_id_pattern, name)]

        if text and len(pq(element).children()) == 0:
            return [text]

        text_elements = []
        for child in pq(element).children():
            text_elements.extend(extract_text(child))

        return text_elements

    # Start with the root element and gather all text elements in a flattened list
    text_elements = extract_text(doc[0])

    return text_elements


def exclude_elements(doc: pq, excludes: List[str]) -> None:
    """
    Removes elements from the document that match the tags in the excludes list.

    :param doc: The PyQuery object representing the HTML document.
    :param excludes: A list of tag names to exclude (e.g., ["style", "script"]).
    """
    for tag in excludes:
        for element in doc(tag):
            pq(element).remove()
