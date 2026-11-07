"""
Robust LLM Agent Playwright inspect_page tool with CLI interface.
Reuses jet.scrapers.browser.playwright_helpers for browser setup.
Uses BeautifulSoup for robust, deterministic, and GENERIC HTML cleaning.

Saves:
1. Full page screenshot (PNG)
2. Cleaned HTML structure (YAML-like accessibility snapshot)
3. Cleaned Raw HTML (pruned, formatted, and strictly attribute-whitelisted)
4. Original Raw HTML (for debugging)

Usage:
    python run_playwright_inspect_page.py https://example.com -m both -c
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Literal, Optional

from bs4 import BeautifulSoup, Comment

# Reuse helpers from the existing module
from jet.scrapers.browser.playwright_helpers import setup_sync_browser_session
from playwright.sync_api import BrowserContext, Page

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Generic Configuration ---

# 1. Strictly Allowed Attribute Names
ALLOWED_ATTR_NAMES = {"class", "id", "role"}

# 2. Generic Value Validation Regex
# Allows: Alphanumeric, underscore, hyphen, hash, dot, colon, slash, whitespace
# Blocks: JavaScript code, JSON, quotes, brackets, equals signs, complex expressions
VALID_VALUE_REGEX = re.compile(r"^[a-zA-Z0-9_\-.#/:\s]+$")

# 3. Generic "Machine-Generated" Identifier Patterns
# These heuristics catch auto-generated classes/IDs from ANY framework without naming them.
AUTO_IDENTIFIER_PATTERNS = [
    re.compile(r"^[a-z]{2,5}-[a-f0-9]{6,}$"),  # e.g., css-abc123, sc-xyz789, jss-123456
    re.compile(r"^_[a-zA-Z0-9]{6,}$"),  # e.g., _abc1234 (common minifier/hash pattern)
    re.compile(r"^[a-f0-9]{8,}$"),  # e.g., 1a2b3c4d (pure hex hashes)
    re.compile(r"^\d+$"),  # e.g., 12345 (pure numeric IDs are often unstable)
    re.compile(r"^\$"),  # e.g., $svelte, $react (internal framework vars)
]


def parse_args():
    """Parse command line arguments with 1-2 char shorthands."""
    parser = argparse.ArgumentParser(
        description="Inspect a webpage and generate LLM-friendly skeletal structure."
    )
    parser.add_argument("url", help="The URL to inspect")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["interactive", "text", "both"],
        default="both",
        help="Content mode: interactive, text, or both (default: both)",
    )
    parser.add_argument(
        "-t",
        "--exclude-tags",
        nargs="+",
        default=["script", "style", "meta", "link", "noscript"],
        help="HTML tags to exclude",
    )
    parser.add_argument(
        "-s",
        "--exclude-selectors",
        nargs="+",
        default=[],
        help="CSS selectors to exclude (e.g., .ad-banner #sidebar)",
    )
    parser.add_argument(
        "-e",
        "--extract-selector",
        nargs="+",
        default=None,
        help="CSS selector(s) to extract. If provided, only these elements are kept.",
    )
    parser.add_argument(
        "-b",
        "--include-boxes",
        action="store_true",
        help="Include bounding box coordinates [box=x,y,w,h] in output",
    )
    parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of DOM tree to traverse",
    )
    parser.add_argument(
        "-c",
        "--clean-html",
        action="store_true",
        help="Remove dynamic/unreliable attributes like auto-generated classes/IDs",
    )
    parser.add_argument(
        "-w",
        "--wait",
        type=int,
        default=2000,
        help="Milliseconds to wait after page load for dynamic content (default: 2000)",
    )
    parser.add_argument(
        "-f",
        "--full-screenshot",
        action="store_true",
        help="Take full-page screenshot instead of viewport only",
    )
    return parser.parse_args()


def is_valid_attr_value(value: str) -> bool:
    """Check if attribute value contains only safe, generic characters."""
    if not value:
        return False
    if isinstance(value, list):
        value = " ".join(value)
    return bool(VALID_VALUE_REGEX.match(value))


def is_machine_generated(identifier: str) -> bool:
    """
    Heuristic check to see if an ID or Class looks auto-generated.
    Returns True if it looks like a hash, minified name, or internal ID.
    """
    if not identifier:
        return False
    return any(pattern.match(identifier) for pattern in AUTO_IDENTIFIER_PATTERNS)


def clean_html_with_bs4(
    html_content: str,
    exclude_tags: list[str],
    exclude_selectors: list[str],
    extract_selectors: Optional[list[str]],
    clean_attrs: bool = True,
) -> str:
    """
    Uses BeautifulSoup to perform all cleaning, extraction, and formatting.
    Implements strict, generic attribute whitelisting and value validation.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Extract Specific Selectors (if provided)
    if extract_selectors:
        # Create a temporary container in the SAME soup to hold extracted elements
        wrapper = soup.new_tag("div")

        for sel in extract_selectors:
            try:
                # select() returns a list of tags
                elements = soup.select(sel)
                for el in elements:
                    # extract() removes the tag from its current position
                    # and returns it, allowing us to append it elsewhere
                    wrapper.append(el.extract())
            except Exception as e:
                print(f"Warning: Selector '{sel}' failed during extraction: {e}")

        # Replace the entire soup body with our wrapper
        # We clear the original soup and insert the wrapper's contents
        soup.clear()
        soup.append(wrapper)

    # 2. Remove Excluded Tags
    for tag_name in exclude_tags:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # 3. Remove Excluded Selectors
    for sel in exclude_selectors:
        try:
            for tag in soup.select(sel):
                tag.decompose()
        except Exception as e:
            pass

    # 4. Remove Comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # 5. Generic Attribute Cleaning
    if clean_attrs:
        for tag in soup.find_all(True):
            attrs_to_keep = {}

            for attr, value in list(tag.attrs.items()):
                # A. Check against strict whitelist names
                is_whitelisted_name = (
                    attr in ALLOWED_ATTR_NAMES
                    or attr.startswith("aria-")
                    or attr.startswith("data-")
                )

                if not is_whitelisted_name:
                    continue

                # B. Validate Value Content (Blocks JS, JSON, Bindings)
                if not is_valid_attr_value(value):
                    continue

                # C. Handle Classes and IDs with Heuristics
                if attr == "class":
                    if isinstance(value, list):
                        # Keep only classes that look human-written
                        cleaned_classes = [
                            cls for cls in value if not is_machine_generated(cls)
                        ]
                        if cleaned_classes:
                            attrs_to_keep[attr] = cleaned_classes
                    continue

                if attr == "id":
                    # Skip IDs that look machine-generated
                    if is_machine_generated(value):
                        continue

                # D. Keep if passed all checks
                attrs_to_keep[attr] = value

            tag.attrs = attrs_to_keep

    # 6. Prettify
    return soup.prettify()


def inspect_page(
    page: Page,
    mode: Literal["interactive", "text", "both"] = "both",
    exclude_tags: Optional[list[str]] = None,
    exclude_selectors: Optional[list[str]] = None,
    include_boxes: bool = False,
    max_depth: Optional[int] = None,
    clean_html: bool = True,
) -> str:
    """
    Inspect page and return minimal skeletal structure for LLM agents.
    """
    if exclude_tags is None:
        exclude_tags = ["script", "style", "meta", "link", "noscript"]
    if exclude_selectors is None:
        exclude_selectors = []

    # Minimal live pruning for aria_snapshot stability
    _prune_dom_live(page, exclude_tags, exclude_selectors)

    options = {
        "mode": "ai",
        "boxes": include_boxes,
    }
    if max_depth is not None:
        options["depth"] = max_depth

    try:
        snapshot = page.locator("body").aria_snapshot(**options)
    except Exception as e:
        print(f"Warning: aria_snapshot failed, using fallback. Error: {e}")
        snapshot = _extract_fallback_dom(page, mode, exclude_tags, exclude_selectors)

    if mode == "interactive":
        snapshot = _filter_interactive_only(snapshot)
    elif mode == "text":
        snapshot = _filter_text_only(snapshot)

    return snapshot


def _prune_dom_live(
    page: Page, exclude_tags: list[str], exclude_selectors: list[str]
) -> None:
    """Minimal pruning on the LIVE DOM to ensure aria_snapshot doesn't see junk."""
    js_code = r"""
    () => {
        const excludedTags = %s;
        excludedTags.forEach(tag => {
            document.querySelectorAll(tag).forEach(el => el.remove());
        });
        const excludedSelectors = %s;
        excludedSelectors.forEach(selector => {
            try {
                document.querySelectorAll(selector).forEach(el => el.remove());
            } catch (e) {}
        });
    }
    """ % (repr(exclude_tags), repr(exclude_selectors))

    try:
        page.evaluate(js_code)
    except Exception:
        pass


def _filter_interactive_only(snapshot: str) -> str:
    """Filter snapshot to show only interactive elements."""
    interactive_roles = {
        "button",
        "link",
        "textbox",
        "checkbox",
        "radio",
        "combobox",
        "listbox",
        "menuitem",
        "tab",
        "switch",
        "searchbox",
        "spinbutton",
        "slider",
        "menuitemcheckbox",
        "menuitemradio",
        "option",
        "treeitem",
    }
    lines = snapshot.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("- text:"):
            continue
        role_match = re.match(r"^\s*-\s+(\w+)", stripped)
        if role_match:
            role = role_match.group(1).lower()
            if role in interactive_roles or role == "generic":
                filtered_lines.append(line)
    return "\n".join(filtered_lines)


def _filter_text_only(snapshot: str) -> str:
    """Filter snapshot to show only text content and semantic structure."""
    content_roles = {
        "heading",
        "paragraph",
        "list",
        "listitem",
        "article",
        "section",
        "navigation",
        "main",
        "complementary",
        "contentinfo",
        "banner",
        "figure",
        "figcaption",
        "table",
        "row",
        "cell",
        "columnheader",
        "rowheader",
        "text",
        "strong",
        "emphasis",
    }
    lines = snapshot.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- text:"):
            filtered_lines.append(line)
            continue
        role_match = re.match(r"^\s*-\s+(\w+)", stripped)
        if role_match:
            role = role_match.group(1).lower()
            if role in content_roles:
                filtered_lines.append(line)
    return "\n".join(filtered_lines)


def _extract_fallback_dom(
    page: Page, mode: str, exclude_tags: list[str], exclude_selectors: list[str]
) -> str:
    """Fallback method: Extract structured DOM when aria_snapshot fails."""
    js_code = r"""
    () => {
        function getElementInfo(el, depth = 0) {
            if (depth > 20) return null;
            const tag = el.tagName.toLowerCase();
            const excludedTags = %s;
            if (excludedTags.includes(tag)) return null;
            
            let name = el.getAttribute('aria-label') || 
                      el.getAttribute('title') ||
                      el.getAttribute('alt') || '';
            
            if (!name) {
                if (el.placeholder) name = el.placeholder;
                else if (tag === 'img' && el.alt) name = el.alt;
            }
            
            let role = el.getAttribute('role') || tag;
            const interactiveTags = ['a', 'button', 'input', 'select', 'textarea'];
            const isInteractive = interactiveTags.includes(tag);
            
            let text = '';
            if (el.childNodes.length === 1 && el.childNodes[0].nodeType === Node.TEXT_NODE) {
                text = el.childNodes[0].textContent.trim().substring(0, 200);
            }
            
            const info = {
                tag: tag, role: role, name: name, text: text, interactive: isInteractive, children: []
            };
            
            for (let child of el.children) {
                const childInfo = getElementInfo(child, depth + 1);
                if (childInfo) info.children.push(childInfo);
            }
            return info;
        }
        return JSON.stringify(getElementInfo(document.body), null, 2);
    }
    """ % repr(exclude_tags)
    try:
        dom_tree = page.evaluate(js_code)
        return _json_to_yaml(dom_tree, mode)
    except Exception:
        return "Error: Could not extract page structure"


def _json_to_yaml(tree_json: str, mode: str) -> str:
    """Convert JSON DOM tree to YAML-like format."""
    import json

    try:
        tree = json.loads(tree_json)
    except:
        return "Error parsing DOM tree"

    def node_to_yaml(node, indent=0):
        if not node:
            return ""
        prefix = "  " * indent
        lines = []
        role = node.get("role", node.get("tag", "unknown"))
        name = node.get("name", "")
        line = f'{prefix}- {role} "{name}"' if name else f"{prefix}- {role}"
        if mode != "text" and node.get("interactive"):
            line += " [interactive]"
        lines.append(line)
        text = node.get("text", "")
        if text:
            lines.append(f'{prefix}  - text: "{text}"')
        for child in node.get("children", []):
            child_yaml = node_to_yaml(child, indent + 1)
            if child_yaml:
                lines.append(child_yaml)
        return "\n".join(lines)

    return node_to_yaml(tree)


def main():
    args = parse_args()

    screenshot_path = OUTPUT_DIR / "screenshot.png"
    html_path = OUTPUT_DIR / "structure.yaml"
    cleaned_html_path = OUTPUT_DIR / "cleaned.html"
    raw_html_path = OUTPUT_DIR / "raw.html"

    print(f"Inspecting: {args.url}")
    print(f"Output directory: {OUTPUT_DIR}")

    context: BrowserContext = setup_sync_browser_session(headless=False)

    try:
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded")
        if args.wait > 0:
            page.wait_for_timeout(args.wait)

        # 1. Screenshot
        page.screenshot(path=str(screenshot_path), full_page=args.full_screenshot)
        print(f"Screenshot saved to: {screenshot_path}")

        # 2. Raw HTML
        raw_html = page.content()
        raw_html_path.write_text(raw_html, encoding="utf-8")
        print(f"Raw HTML saved to: {raw_html_path}")

        # 3. Cleaned HTML (Using BS4)
        cleaned_html = clean_html_with_bs4(
            raw_html,
            args.exclude_tags,
            args.exclude_selectors,
            args.extract_selector,
            args.clean_html,
        )
        cleaned_html_path.write_text(cleaned_html, encoding="utf-8")
        print(f"Cleaned HTML saved to: {cleaned_html_path}")

        # 4. Structure (YAML)
        structure = inspect_page(
            page=page,
            mode=args.mode,
            exclude_tags=args.exclude_tags,
            exclude_selectors=args.exclude_selectors,
            include_boxes=args.include_boxes,
            max_depth=args.max_depth,
            clean_html=args.clean_html,
        )
        html_path.write_text(structure, encoding="utf-8")
        print(f"Structure saved to: {html_path}")
        print(f"Structure size: {len(structure)} chars (~{len(structure) // 4} tokens)")

    except Exception as e:
        print(f"Error during inspection: {e}")
        raise
    finally:
        try:
            context.close()
        except:
            pass

    print("\nDone! Files generated in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
