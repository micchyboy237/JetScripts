#!/usr/bin/env python3
"""
Enhanced Multi-URL Cloudflare CAPTCHA Bypass Scraper

Features:
- Processes multiple URLs in a single browser session (same tab).
- Organizes output into subdirectories: pages/url_<index>/
- Incorporates advanced techniques from D3-vin/Turnstile-Solver-NEW:
  - Route interception for faster Turnstile solving
  - cf_clearance cookie harvesting for IUAM challenges
  - Human-like iframe clicking
  - Comprehensive Cloudflare detection markers
- Uses Mattwmaster58/playwright_stealth recommended API (Stealth context manager)

Requirements:
    pip install playwright playwright-stealth
    playwright install chromium
"""

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

from playwright.async_api import Frame, Page, async_playwright
from playwright_stealth import Stealth

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# JavaScript Templates & Snippets (from D3-vin/Turnstile-Solver-NEW)
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Turnstile</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a1a; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
</style>
</head>
<body>
<!--WIDGET-->
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
</body></html>"""

WIDGET_INJECT_JS = """(k) => {
    const d = document.createElement('div');
    d.className = 'cf-turnstile';
    d.setAttribute('data-sitekey', k);
    document.body.prepend(d);
}"""

TOKEN_VALUE_JS = """() => { 
    const e = document.querySelector('[name=cf-turnstile-response]');
    return e ? e.value : '' 
}"""

CF_MARKERS_JS = """() => {
  const html = document.documentElement.outerHTML;
  const t = (document.title || '').toLowerCase();
  return (
    !!document.querySelector('#challenge-form, form#challenge-form') ||
    !!document.querySelector('#cf-wrapper, .cf-browser-verification, #challenge-running, #trk_jschal_js') ||
    !!document.querySelector('iframe[src*="challenges.cloudflare.com"]') ||
    /window\\._cf_chl_opt|__cf_chl_/.test(html) ||
    /just a moment|attention required|checking your browser|verifying you are human/.test(t)
  );
}"""

REPLAY_WARNING = (
    "cf_clearance is bound to IP + JA3/TLS + User-Agent. Replay only from the "
    "same IP, with this exact User-Agent, over a TLS stack producing the same JA3."
)


class CloudflareDetector:
    """Enhanced Cloudflare CAPTCHA detector with comprehensive markers."""

    CLOUDFLARE_INTERSTITIAL_URLS = [
        "challenges.cloudflare.com",
        "/cdn-cgi/challenge-platform/",
    ]

    TURNSTILE_SELECTORS = [
        "[data-sitekey]",
        ".cf-turnstile",
        '[class*="turnstile"]',
        'iframe[src*="challenges.cloudflare.com"]',
    ]

    @staticmethod
    async def is_cloudflare_interstitial(page: Page) -> bool:
        """Check if page is a Cloudflare interstitial challenge using JS markers."""
        try:
            current_url = page.url
            if any(
                url in current_url
                for url in CloudflareDetector.CLOUDFLARE_INTERSTITIAL_URLS
            ):
                return True

            # Use comprehensive JS detection
            cf_detected = await page.evaluate(CF_MARKERS_JS)
            if cf_detected:
                return True

            # Check for cfParams which indicates interstitial challenge
            cf_params = await page.evaluate("window.cfParams")
            if cf_params:
                return True

            return False
        except Exception as e:
            logger.debug(f"Error checking for interstitial: {e}")
            return False

    @staticmethod
    async def has_turnstile_widget(queryable: Union[Page, Frame]) -> bool:
        """Check if page contains a Cloudflare Turnstile widget."""
        try:
            for selector in CloudflareDetector.TURNSTILE_SELECTORS:
                elements = queryable.locator(selector)
                if await elements.count() > 0:
                    return True
            return False
        except Exception as e:
            logger.debug(f"Error checking for turnstile: {e}")
            return False

    @staticmethod
    async def detect_challenge_type(page: Page) -> Optional[str]:
        """Detect the type of Cloudflare challenge present."""
        if await CloudflareDetector.is_cloudflare_interstitial(page):
            return "interstitial"

        # Check all frames for turnstile widgets
        for frame in page.frames:
            if await CloudflareDetector.has_turnstile_widget(frame):
                return "turnstile"

        return None

    @staticmethod
    def extract_sitekey_from_page(page: Page) -> Optional[str]:
        """Extract Turnstile sitekey from page."""
        try:
            sitekey_el = page.locator("[data-sitekey]").first
            if sitekey_el:
                return await sitekey_el.get_attribute("data-sitekey")
        except Exception:
            pass
        return None


class TurnstileSolver:
    """Advanced Turnstile solver with route interception and real-page fallback."""

    def __init__(self, max_attempts: int = 25, timeout: int = 60):
        self.max_attempts = max_attempts
        self.timeout = timeout

    @staticmethod
    def route_glob(url: str) -> str:
        """Convert URL to glob pattern for route interception."""
        parts = urlsplit(url)
        if parts.path in ("", "/"):
            return f"{parts.scheme}://{parts.netloc}/**"
        return url

    @staticmethod
    def build_route_html(
        sitekey: str, action: Optional[str] = None, cdata: Optional[str] = None
    ) -> str:
        """Build minimal HTML page with Turnstile widget."""
        widget_html = f'<div class="cf-turnstile" data-sitekey="{sitekey}"'
        if action:
            widget_html += f' data-action="{action}"'
        if cdata:
            widget_html += f' data-cdata="{cdata}"'
        widget_html += "></div>"

        return HTML_TEMPLATE.replace("<!--WIDGET-->", widget_html)

    async def _human_click_iframe(self, page: Page, frame: Frame) -> bool:
        """Click on iframe at human-like coordinates."""
        try:
            el = await frame.frame_element()
            box = await el.bounding_box()
        except Exception:
            return False

        if not box or box["width"] < 20:
            return False

        # Click at center-left of iframe (more human-like)
        await page.mouse.click(box["x"] + 30, box["y"] + box["height"] / 2)
        logger.debug(
            f"Clicked iframe at ({box['x'] + 30}, {box['y'] + box['height'] / 2})"
        )
        return True

    async def click_turnstile_checkbox(self, page: Page, attempts: int = 25) -> bool:
        """Click Turnstile checkbox with human-like behavior."""
        for attempt in range(attempts):
            for fr in page.frames:
                if "challenges.cloudflare.com" not in (fr.url or ""):
                    continue

                # Try human-like iframe click first
                if await self._human_click_iframe(page, fr):
                    logger.info("Clicked Turnstile via human-like iframe interaction")
                    return True

                # Fallback to direct element clicking
                for sel in ("input[type=checkbox]", "label", "body"):
                    try:
                        await fr.click(sel, timeout=2000)
                        logger.info(f"Clicked Turnstile via selector: {sel}")
                        return True
                    except Exception:
                        continue

            await asyncio.sleep(1)

        logger.warning("Failed to click Turnstile checkbox")
        return False

    async def _get_token_route(
        self, page: Page, max_attempts: int = 20
    ) -> Optional[str]:
        """Extract token from route-intercepted page."""
        for _ in range(max_attempts):
            try:
                val = await page.input_value("[name=cf-turnstile-response]")
                if val == "":
                    try:
                        await page.click('//div[@class="cf-turnstile"]', timeout=3000)
                    except Exception:
                        pass
                    await asyncio.sleep(1)
                else:
                    el = await page.query_selector("[name=cf-turnstile-response]")
                    return await el.get_attribute("value") if el else None
            except Exception:
                await asyncio.sleep(1)
        return None

    async def solve_route(
        self,
        page: Page,
        url: str,
        sitekey: str,
        action: Optional[str] = None,
        cdata: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        """Solve Turnstile via route interception (fast method)."""
        page_data = self.build_route_html(sitekey, action, cdata)
        pattern = self.route_glob(url)

        # Clear previous routes to avoid conflicts
        await page.unroute_all()

        await page.route(
            pattern,
            lambda route: route.fulfill(
                body=page_data, status=200, content_type="text/html"
            ),
        )

        logger.info(f"Route-intercept enabled for: {pattern}")
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        token = await self._get_token_route(page)
        return token, pattern

    async def solve_realpage(
        self, page: Page, url: str, sitekey: str, timeout_s: int = 60
    ) -> Optional[str]:
        """Solve Turnstile on real page (fallback method)."""
        logger.info(f"Solving on real page: {url}")

        # Clear routes if any were set previously
        await page.unroute_all()

        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(2)

        # Inject widget if sitekey provided
        if sitekey:
            await page.evaluate(WIDGET_INJECT_JS, sitekey)
            await asyncio.sleep(3)

        # Click the checkbox
        await self.click_turnstile_checkbox(page)

        # Poll for token
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                token = await page.evaluate(TOKEN_VALUE_JS)
            except Exception:
                token = ""

            if token:
                logger.info(f"Got Turnstile token: {token[:20]}...")
                return token

            await asyncio.sleep(1)

        logger.warning("Timeout waiting for Turnstile token")
        return None

    async def solve(
        self,
        page: Page,
        url: str,
        sitekey: str,
        action: Optional[str] = None,
        cdata: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        """
        Solve Turnstile with route-intercept first, real-page fallback.
        """
        # Try route interception first (faster)
        token, pattern = await self.solve_route(page, url, sitekey, action, cdata)

        if token:
            logger.info("Turnstile solved via route interception")
            return token, "route"

        # Fallback to real page
        logger.info("Route interception failed, trying real-page method")
        token = await self.solve_realpage(page, url, sitekey)
        return token, "real-page"


class ClearanceHarvester:
    """Harvests cf_clearance cookies from IUAM challenges."""

    @staticmethod
    def find_clearance(cookies: List[Dict]) -> Optional[Dict]:
        """Find cf_clearance cookie in cookie list."""
        return next((c for c in cookies if c.get("name") == "cf_clearance"), None)

    async def is_interstitial(self, page: Page) -> bool:
        """Check if page still shows interstitial challenge."""
        try:
            return bool(await page.evaluate(CF_MARKERS_JS))
        except Exception:
            return False

    async def wait_clearance(
        self, page: Page, timeout_s: int = 60
    ) -> Tuple[Optional[Dict], List]:
        """Wait for cf_clearance cookie to be set."""
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            cookies = await page.context.cookies()
            clearance = self.find_clearance(cookies)

            if clearance and not await self.is_interstitial(page):
                logger.info("cf_clearance cookie obtained")
                return clearance, cookies

            if clearance:
                # Cookie exists but challenge may still be processing
                await asyncio.sleep(1)
                cookies = await page.context.cookies()
                return self.find_clearance(cookies), cookies

            await asyncio.sleep(1)

        logger.warning("Timeout waiting for cf_clearance")
        return None, await page.context.cookies()

    async def solve(self, page: Page, url: str, timeout_s: int = 60) -> Dict:
        """Pass Cloudflare interstitial and harvest cf_clearance."""
        start = time.monotonic()

        # Clear routes
        await page.unroute_all()

        await page.goto(url, wait_until="domcontentloaded", timeout=45000)

        # Try to click Turnstile if present
        turnstile_solver = TurnstileSolver()
        try:
            await turnstile_solver.click_turnstile_checkbox(page, attempts=8)
        except Exception:
            pass

        # Wait for clearance
        clearance, cookies = await self.wait_clearance(page, timeout_s)

        # Get browser info
        ua = await page.evaluate("() => navigator.userAgent")
        lang = await page.evaluate("() => navigator.language")

        result = {
            "cf_clearance": clearance,
            "cookies": cookies,
            "user_agent": ua,
            "headers": {
                "User-Agent": ua,
                "Accept-Language": lang,
            },
            "elapsed": round(time.monotonic() - start, 1),
            "warning": REPLAY_WARNING,
        }

        if not clearance:
            result["error"] = "cf_clearance not set (challenge unsolved)"
            logger.error("Failed to obtain cf_clearance cookie")
        else:
            logger.info(f"cf_clearance obtained in {result['elapsed']}s")

        return result


class StealthConfigurator:
    """Enhanced stealth configuration with sec-ch-ua headers."""

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    SEC_CH_UA = '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"'

    @staticmethod
    def get_random_user_agent() -> str:
        return random.choice(StealthConfigurator.USER_AGENTS)

    @staticmethod
    def get_browser_args() -> Dict[str, Any]:
        return {
            "headless": False,
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-accelerated-2d-canvas",
                "--disable-gpu",
                "--window-size=1920,1080",
            ],
        }

    @staticmethod
    def get_context_options(
        user_agent: str, proxy: Optional[str] = None
    ) -> Dict[str, Any]:
        opts = {
            "user_agent": user_agent,
            "viewport": {"width": 1920, "height": 1080},
            "extra_http_headers": {
                "sec-ch-ua": StealthConfigurator.SEC_CH_UA,
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
            },
        }

        if proxy:
            opts["proxy"] = StealthConfigurator.parse_proxy(proxy)

        return opts

    @staticmethod
    def parse_proxy(proxy: str) -> Dict[str, str]:
        if "://" not in proxy:
            proxy = f"http://{proxy}"

        scheme, rest = proxy.split("://", 1)

        if "@" in rest:
            auth, addr = rest.split("@", 1)
            user, pwd = auth.split(":", 1)
            return {"server": f"{scheme}://{addr}", "username": user, "password": pwd}

        return {"server": proxy}


class HumanBehaviorSimulator:
    @staticmethod
    async def simulate_human_behavior(page: Page) -> None:
        logger.debug("Simulating human behavior...")
        for _ in range(3):
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            await page.mouse.move(x, y, steps=random.randint(5, 15))
            await asyncio.sleep(random.uniform(0.1, 0.3))

        await page.evaluate("window.scrollBy(0, window.innerHeight * 0.3)")
        await asyncio.sleep(random.uniform(0.5, 1.0))
        await page.evaluate("window.scrollBy(0, -window.innerHeight * 0.1)")
        await asyncio.sleep(random.uniform(0.3, 0.6))


class ScreenshotManager:
    @staticmethod
    async def capture_aria_snapshot(page: Page, output_path: str) -> str:
        try:
            snapshot = await page.aria_snapshot()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(snapshot)
            logger.info(f"ARIA snapshot saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to capture ARIA snapshot: {e}")
            raise

    @staticmethod
    async def capture_full_screenshot(page: Page, output_path: str) -> str:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            await page.screenshot(path=output_path, full_page=True)
            logger.info(f"Full-page screenshot saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            raise


class CloudflareScraper:
    """Enhanced scraper supporting multiple URLs in same tab."""

    def __init__(
        self,
        output_dir: str = "./output",
        max_retries: int = 3,
        timeout: int = 60000,
        enable_stealth: bool = True,
        simulate_human: bool = True,
        use_route_intercept: bool = True,
        proxy: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_stealth = enable_stealth
        self.simulate_human = simulate_human
        self.use_route_intercept = use_route_intercept
        self.proxy = proxy

        self.detector = CloudflareDetector()
        self.turnstile_solver = TurnstileSolver(max_attempts=max_retries)
        self.clearance_harvester = ClearanceHarvester()
        self.stealth_config = StealthConfigurator()
        self.behavior_simulator = HumanBehaviorSimulator()
        self.screenshot_mgr = ScreenshotManager()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_url_output_dir(self, base_dir: Path, index: int, url: str) -> Path:
        """Create subdir pages/url_<num>/"""
        url_dir = base_dir / "pages" / f"url_{index}"
        url_dir.mkdir(parents=True, exist_ok=True)
        return url_dir

    def _generate_filenames(self, url_dir: Path, url: str) -> Dict[str, str]:
        """Generate output filenames inside the specific url directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract domain for filename prefix to keep it readable
        domain = url.split("//")[-1].split("/")[0].replace(".", "_")

        return {
            "screenshot": str(url_dir / f"{domain}_{timestamp}_screenshot.png"),
            "aria_snapshot": str(url_dir / f"{domain}_{timestamp}_aria_snapshot.yaml"),
            "session_bundle": str(url_dir / f"{domain}_{timestamp}_session.json"),
        }

    async def _handle_cloudflare_challenge(self, page: Page, url: str) -> Dict:
        """Detect and handle Cloudflare challenges."""
        # Wait for initial page load
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            logger.warning("Initial page load timeout, continuing anyway")

        # Detect challenge type
        challenge_type = await self.detector.detect_challenge_type(page)

        if not challenge_type:
            logger.info("No Cloudflare challenge detected")
            return {"status": "no_challenge"}

        logger.info(f"Detected Cloudflare {challenge_type} challenge")

        if challenge_type == "turnstile":
            # Extract sitekey
            sitekey = self.detector.extract_sitekey_from_page(page)
            if not sitekey:
                logger.warning("Could not extract Turnstile sitekey")
                return {"status": "failed", "error": "No sitekey found"}

            # Solve Turnstile
            if self.use_route_intercept:
                token, method = await self.turnstile_solver.solve(page, url, sitekey)
            else:
                token = await self.turnstile_solver.solve_realpage(page, url, sitekey)
                method = "real-page-only"

            if token:
                logger.info(f"Turnstile solved via {method}")
                return {
                    "status": "solved",
                    "type": "turnstile",
                    "token": token,
                    "method": method,
                }
            else:
                return {"status": "failed", "error": "Turnstile solving failed"}

        elif challenge_type == "interstitial":
            # Harvest cf_clearance
            bundle = await self.clearance_harvester.solve(page, url)

            if bundle.get("cf_clearance"):
                logger.info("cf_clearance obtained successfully")
                return {"status": "solved", "type": "interstitial", "bundle": bundle}
            else:
                return {"status": "failed", "error": "cf_clearance not obtained"}

        return {"status": "unknown"}

    async def scrape_single_url(
        self, page: Page, url: str, index: int
    ) -> Dict[str, str]:
        """Scrape a single URL within the existing page context."""
        logger.info(f"[{index}] Starting scrape of: {url}")

        # Create specific output directory for this URL
        url_output_dir = self._get_url_output_dir(self.output_dir, index, url)
        filenames = self._generate_filenames(url_output_dir, url)

        try:
            # Navigate to URL
            logger.info(f"[{index}] Navigating to: {url}")
            await page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")

            # Simulate human behavior
            if self.simulate_human:
                await self.behavior_simulator.simulate_human_behavior(page)

            # Handle Cloudflare challenges
            challenge_result = await self._handle_cloudflare_challenge(page, url)

            if challenge_result["status"] == "failed":
                logger.warning(
                    f"[{index}] Challenge handling failed: {challenge_result.get('error')}"
                )

            # Save session bundle if obtained
            if "bundle" in challenge_result:
                session_file = filenames["session_bundle"]
                with open(session_file, "w") as f:
                    json.dump(challenge_result["bundle"], f, indent=2, default=str)
                logger.info(f"[{index}] Session bundle saved to: {session_file}")

            # Wait for dynamic content
            await asyncio.sleep(2)

            # Capture outputs
            await self.screenshot_mgr.capture_aria_snapshot(
                page, filenames["aria_snapshot"]
            )

            await self.screenshot_mgr.capture_full_screenshot(
                page, filenames["screenshot"]
            )

            logger.info(f"[{index}] Scraping completed successfully")
            return filenames

        except Exception as e:
            logger.error(f"[{index}] Error during scraping: {e}", exc_info=True)
            raise

    async def scrape(self, urls: List[str]) -> List[Dict[str, str]]:
        """Scrape multiple URLs in the same browser session/tab."""
        results = []

        # Use the recommended Stealth context manager from Mattwmaster58/playwright_stealth
        # This ensures all pages created within this block are stealthy
        stealth = Stealth()

        async with stealth.use_async(async_playwright()) as p:
            # Launch browser
            browser_args = self.stealth_config.get_browser_args()
            browser = await p.chromium.launch(**browser_args)

            # Create context with enhanced options
            user_agent = self.stealth_config.get_random_user_agent()
            context_opts = self.stealth_config.get_context_options(
                user_agent, self.proxy
            )
            context = await browser.new_context(**context_opts)

            # Create a SINGLE page to reuse
            page = await context.new_page()

            try:
                # Iterate through URLs
                # Note: Stealth is already applied by the context manager,
                # so no need to call apply_stealth(page) manually.
                for i, url in enumerate(urls, start=1):
                    try:
                        result = await self.scrape_single_url(page, url, i)
                        results.append(result)

                        # Small delay between URLs to be polite and avoid rate limits
                        if i < len(urls):
                            logger.info(f"Waiting 2 seconds before next URL...")
                            await asyncio.sleep(2)

                    except Exception as e:
                        logger.error(f"[{i}] Failed to process {url}: {e}")
                        # Append error result or skip? Let's append a placeholder
                        results.append({"error": str(e), "url": url})

            finally:
                await browser.close()

        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-URL Cloudflare CAPTCHA Bypass Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://example.com https://example.org
  %(prog)s --urls https://example.com https://example.org --output-dir ./results
  %(prog)s https://example.com --proxy http://user:pass@proxy:8080
        """,
    )

    parser.add_argument("urls", nargs="+", help="One or more URLs to scrape")

    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(OUTPUT_DIR),
        help="Base output directory",
    )

    parser.add_argument(
        "--max-retries",
        "-r",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)",
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=60000,
        help="Page load timeout in ms (default: 60000)",
    )

    parser.add_argument(
        "--no-stealth", action="store_true", help="Disable stealth mode"
    )

    parser.add_argument(
        "--no-human-behavior",
        action="store_true",
        help="Disable human behavior simulation",
    )

    parser.add_argument(
        "--no-route-intercept",
        action="store_true",
        help="Disable route interception (use real-page only)",
    )

    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Proxy server (e.g., http://user:pass@host:port)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate URLs
    valid_urls = []
    for url in args.urls:
        if url.startswith(("http://", "https://")):
            valid_urls.append(url)
        else:
            logger.error(f"Skipping invalid URL (must start with http/https): {url}")

    if not valid_urls:
        logger.error("No valid URLs provided.")
        return 1

    scraper = CloudflareScraper(
        output_dir=args.output_dir,
        max_retries=args.max_retries,
        timeout=args.timeout,
        enable_stealth=not args.no_stealth,
        simulate_human=not args.no_human_behavior,
        use_route_intercept=not args.no_route_intercept,
        proxy=args.proxy,
    )

    try:
        results = await scraper.scrape(valid_urls)

        print("\n" + "=" * 60)
        print("Scraping Complete!")
        print("=" * 60)
        for i, res in enumerate(results, start=1):
            if "error" in res:
                print(f"URL {i}: FAILED - {res['error']}")
            else:
                print(f"URL {i} Results:")
                print(f"  Screenshot: {res['screenshot']}")
                print(f"  ARIA Snapshot: {res['aria_snapshot']}")
                if "session_bundle" in res:
                    print(f"  Session Bundle: {res['session_bundle']}")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Scraping failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
