#!/usr/bin/env python3
"""
Cloudflare CAPTCHA Bypass Scraper with Playwright (Multi-URL Version)

Updated to use Mattwmaster58/playwright_stealth recommended API:
- Uses Stealth() context manager or context-level application
- Processes multiple URLs in a single browser session (same tab)
- Organizes output into pages/url_<index>/ subdirectories

Requirements:
    pip install playwright playwright-stealth
    playwright install chromium
"""

import argparse
import asyncio
import logging
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from playwright.async_api import Frame, Page, async_playwright
from playwright_stealth import Stealth  # Updated import

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CloudflareDetector:
    """Detects Cloudflare CAPTCHA challenges on a page."""

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
        """Check if page is a Cloudflare interstitial challenge."""
        try:
            current_url = page.url
            if any(
                url in current_url
                for url in CloudflareDetector.CLOUDFLARE_INTERSTITIAL_URLS
            ):
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


class CloudflareSolver:
    """Solves Cloudflare CAPTCHA challenges automatically."""

    def __init__(self, max_attempts: int = 3, wait_timeout: int = 30):
        self.max_attempts = max_attempts
        self.wait_timeout = wait_timeout

    async def solve(self, page: Page, challenge_type: str) -> bool:
        """
        Solve Cloudflare challenge based on type.

        Args:
            page: Playwright page object
            challenge_type: Type of challenge ('interstitial' or 'turnstile')

        Returns:
            True if solved successfully, False otherwise
        """
        logger.info(f"Attempting to solve Cloudflare {challenge_type} challenge...")

        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"Solving attempt {attempt}/{self.max_attempts}")

            try:
                if challenge_type == "interstitial":
                    success = await self._solve_interstitial(page)
                elif challenge_type == "turnstile":
                    success = await self._solve_turnstile(page)
                else:
                    logger.error(f"Unknown challenge type: {challenge_type}")
                    return False

                if success:
                    logger.info(f"Successfully solved {challenge_type} challenge")
                    return True

            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < self.max_attempts:
                    await page.reload(wait_until="networkidle")
                    await asyncio.sleep(2)

        logger.error(
            f"Failed to solve {challenge_type} challenge after {self.max_attempts} attempts"
        )
        return False

    async def _solve_interstitial(self, page: Page) -> bool:
        """Solve Cloudflare interstitial challenge by waiting for auto-resolution."""
        start_time = time.time()

        while time.time() - start_time < self.wait_timeout:
            # Check if challenge is still present
            if not await CloudflareDetector.is_cloudflare_interstitial(page):
                return True

            # Try to click the verify button if present
            try:
                verify_button = page.locator(
                    'button:has-text("Verify"), input[type="submit"]'
                )
                if await verify_button.count() > 0:
                    await verify_button.first.click()
                    await asyncio.sleep(2)
            except Exception:
                pass

            await asyncio.sleep(1)

        return False

    async def _solve_turnstile(self, page: Page) -> bool:
        """Solve Cloudflare Turnstile challenge by clicking the checkbox."""
        # Find Cloudflare iframe
        cf_iframe = None
        for frame in page.frames:
            if "challenges.cloudflare.com" in frame.url:
                cf_iframe = frame
                break

        if not cf_iframe:
            logger.warning("Cloudflare iframe not found")
            return False

        # Wait for checkbox to appear and be clickable
        try:
            checkbox = cf_iframe.locator('input[type="checkbox"], .cf-turnstile input')
            await checkbox.wait_for(state="visible", timeout=10000)

            # Click the checkbox
            await checkbox.first.click()
            logger.info("Clicked Turnstile checkbox")

            # Wait for success indicator
            await asyncio.sleep(3)

            # Check if success element appeared
            success_elements = cf_iframe.locator('#success, [aria-label*="success"]')
            if await success_elements.count() > 0:
                return True

            # Alternative: check if main page content is now accessible
            await asyncio.sleep(2)
            return not await CloudflareDetector.has_turnstile_widget(page)

        except Exception as e:
            logger.warning(f"Error solving turnstile: {e}")
            return False


class StealthConfigurator:
    """Configures browser for stealth mode using Mattwmaster58/playwright_stealth."""

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    @staticmethod
    def get_random_user_agent() -> str:
        """Get a random user agent string."""
        return random.choice(StealthConfigurator.USER_AGENTS)

    @staticmethod
    def get_browser_args() -> Dict[str, Any]:
        """Get browser launch arguments for stealth."""
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


class HumanBehaviorSimulator:
    """Simulates human-like browser behavior to avoid detection."""

    @staticmethod
    async def simulate_human_behavior(page: Page) -> None:
        """Simulate realistic human interactions."""
        logger.debug("Simulating human behavior...")

        # Random mouse movements
        for _ in range(3):
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            await page.mouse.move(x, y, steps=random.randint(5, 15))
            await asyncio.sleep(random.uniform(0.1, 0.3))

        # Scroll down slightly
        await page.evaluate("window.scrollBy(0, window.innerHeight * 0.3)")
        await asyncio.sleep(random.uniform(0.5, 1.0))

        # Scroll back up slightly
        await page.evaluate("window.scrollBy(0, -window.innerHeight * 0.1)")
        await asyncio.sleep(random.uniform(0.3, 0.6))


class ScreenshotManager:
    """Manages screenshot and snapshot capture."""

    @staticmethod
    async def capture_aria_snapshot(page: Page, output_path: str) -> str:
        """
        Capture ARIA snapshot (accessibility tree) as YAML.
        """
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
        """
        Capture full-page screenshot.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            await page.screenshot(path=output_path, full_page=True)

            logger.info(f"Full-page screenshot saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            raise


class CloudflareScraper:
    """Main scraper class that orchestrates the entire process."""

    def __init__(
        self,
        output_dir: str = "./output",
        max_retries: int = 3,
        timeout: int = 60000,
        enable_stealth: bool = True,
        simulate_human: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_stealth = enable_stealth
        self.simulate_human = simulate_human

        self.detector = CloudflareDetector()
        self.solver = CloudflareSolver(max_attempts=max_retries)
        self.stealth_config = StealthConfigurator()
        self.behavior_simulator = HumanBehaviorSimulator()
        self.screenshot_mgr = ScreenshotManager()

        # Create base output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_url_output_dir(self, index: int) -> Path:
        """Create subdir pages/url_<num>/"""
        url_dir = self.output_dir / "pages" / f"url_{index}"
        url_dir.mkdir(parents=True, exist_ok=True)
        return url_dir

    def _generate_filenames(self, url_dir: Path, url: str) -> Dict[str, str]:
        """Generate output filenames based on URL and timestamp inside the specific dir."""
        # Extract domain from URL for filename readability
        domain = url.split("//")[-1].split("/")[0].replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return {
            "screenshot": str(url_dir / f"{domain}_{timestamp}_screenshot.png"),
            "aria_snapshot": str(url_dir / f"{domain}_{timestamp}_aria_snapshot.yaml"),
        }

    async def _handle_cloudflare_challenge(self, page: Page) -> bool:
        """
        Detect and handle Cloudflare challenges.
        """
        # Wait for initial page load
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            logger.warning("Initial page load timeout, continuing anyway")

        # Detect challenge type
        challenge_type = await self.detector.detect_challenge_type(page)

        if not challenge_type:
            logger.info("No Cloudflare challenge detected")
            return True

        logger.info(f"Detected Cloudflare {challenge_type} challenge")

        # Attempt to solve
        solved = await self.solver.solve(page, challenge_type)

        if solved:
            # Wait for actual page to load after solving
            try:
                await page.wait_for_load_state("networkidle", timeout=30000)
                logger.info("Page loaded successfully after solving challenge")
            except Exception:
                logger.warning("Timeout waiting for page load after solving")
            return True
        else:
            logger.error("Failed to solve Cloudflare challenge")
            return False

    async def scrape_single_url(
        self, page: Page, url: str, index: int
    ) -> Dict[str, str]:
        """Scrape a single URL using the provided page object."""
        logger.info(f"[{index}] Starting scrape of: {url}")

        # Create specific output directory for this URL
        url_output_dir = self._get_url_output_dir(index)
        filenames = self._generate_filenames(url_output_dir, url)

        try:
            # Navigate to URL
            logger.info(f"[{index}] Navigating to: {url}")
            await page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")

            # Simulate human behavior if enabled
            if self.simulate_human:
                await self.behavior_simulator.simulate_human_behavior(page)

            # Handle Cloudflare challenges
            challenge_handled = await self._handle_cloudflare_challenge(page)

            if not challenge_handled:
                logger.warning(
                    f"[{index}] Continuing despite failed challenge handling"
                )

            # Wait a bit more for dynamic content
            await asyncio.sleep(2)

            # Capture ARIA snapshot
            await self.screenshot_mgr.capture_aria_snapshot(
                page, filenames["aria_snapshot"]
            )

            # Capture full-page screenshot
            await self.screenshot_mgr.capture_full_screenshot(
                page, filenames["screenshot"]
            )

            logger.info(f"[{index}] Scraping completed successfully")
            return filenames

        except Exception as e:
            logger.error(f"[{index}] Error during scraping: {e}", exc_info=True)
            raise

    async def scrape(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Scrape multiple URLs, handling Cloudflare challenges automatically.
        Reuses the same browser tab for efficiency.
        """
        results = []

        # Use the recommended Stealth context manager
        # This ensures all pages created within this block are stealthy
        stealth = Stealth()

        async with stealth.use_async(async_playwright()) as p:
            # Launch browser with stealth configuration
            browser_args = self.stealth_config.get_browser_args()
            browser = await p.chromium.launch(**browser_args)

            # Create context with random user agent
            user_agent = self.stealth_config.get_random_user_agent()
            context = await browser.new_context(
                user_agent=user_agent,
                viewport={"width": 1920, "height": 1080},
            )

            # Create a SINGLE page to reuse for all URLs
            page = await context.new_page()

            try:
                # Iterate through URLs
                for i, url in enumerate(urls, start=1):
                    try:
                        result = await self.scrape_single_url(page, url, i)
                        results.append(result)

                        # Small delay between URLs to be polite and allow cleanup
                        if i < len(urls):
                            logger.info(f"Waiting 2 seconds before next URL...")
                            await asyncio.sleep(2)

                    except Exception as e:
                        logger.error(f"[{i}] Failed to process {url}: {e}")
                        # Append error placeholder so indices match
                        results.append({"error": str(e), "url": url})

            finally:
                await browser.close()

        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cloudflare CAPTCHA Bypass Scraper with Playwright (Multi-URL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://example.com
  %(prog)s https://example.com https://example.org https://example.net
  %(prog)s https://example.com --output-dir ./results --max-retries 5
  %(prog)s https://example.com --no-stealth --no-human-behavior
        """,
    )

    parser.add_argument("urls", nargs="+", help="One or more URLs to scrape")

    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(OUTPUT_DIR),
        help="Base directory to save screenshots and snapshots",
    )

    parser.add_argument(
        "--max-retries",
        "-r",
        type=int,
        default=3,
        help="Maximum number of retry attempts for CAPTCHA solving (default: 3)",
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=60000,
        help="Page load timeout in milliseconds (default: 60000)",
    )

    parser.add_argument(
        "--no-stealth",
        action="store_true",
        help="Disable stealth mode (not recommended for Cloudflare sites)",
    )

    parser.add_argument(
        "--no-human-behavior",
        action="store_true",
        help="Disable human behavior simulation",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose/debug logging"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging level
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

    # Create scraper instance
    scraper = CloudflareScraper(
        output_dir=args.output_dir,
        max_retries=args.max_retries,
        timeout=args.timeout,
        enable_stealth=not args.no_stealth,
        simulate_human=not args.no_human_behavior,
    )

    try:
        # Run scraper
        results = await scraper.scrape(valid_urls)

        print("\n" + "=" * 60)
        print("Scraping Complete!")
        print("=" * 60)

        for i, result in enumerate(results, start=1):
            if "error" in result:
                print(
                    f"URL {i} ({result.get('url', 'Unknown')}): FAILED - {result['error']}"
                )
            else:
                print(f"URL {i} Results:")
                print(f"  Screenshot: {result['screenshot']}")
                print(f"  ARIA Snapshot: {result['aria_snapshot']}")

        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Scraping failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
