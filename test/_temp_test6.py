import difflib
import os
import platform
import random
import time
import traceback
from pathlib import Path
from time import sleep
from typing import Tuple

from fake_useragent import UserAgent
from playwright.sync_api import TimeoutError, sync_playwright
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.traceback import install

console = Console()
install()  # makes tracebacks much more readable in terminal


def get_chrome_paths() -> Tuple[str, str, str]:
    system = platform.system()
    home = Path.home()

    if system == "Darwin":
        base = home / "Library/Application Support/Google/Chrome"
        os_short = "mac"
    elif system in ("Windows", "win32", "cygwin"):
        base = home / "AppData/Local/Google/Chrome/User Data"
        os_short = "win"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    profile_name = "Default"  # ← change if you're using a different profile
    resolved = base.resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"Chrome profile directory missing:\n {resolved}\n"
            "Open Chrome at least once first."
        )

    return str(resolved), profile_name, os_short


def get_random_ua(os_short: str) -> str:
    ua_gen = UserAgent()
    if os_short in ("mac", "win"):
        return ua_gen.chrome or ua_gen.random
    return ua_gen.random


def main():
    user_data_dir, profile_name, os_short = get_chrome_paths()
    user_agent = get_random_ua(os_short)

    # ── Configuration panel ──────────────────────────────────────────
    config = (
        f"[b]OS[/b]             : {platform.system()}\n"
        f"[b]User Data Dir[/b]   : {user_data_dir}\n"
        f"[b]Profile[/b]         : {profile_name}\n"
        f"[b]Random UA[/b]       : {user_agent}"
    )
    console.print(
        Panel.fit(
            config, title="Configuration", border_style="bright_blue", padding=(1, 2)
        )
    )

    console.print("\n[bold yellow]Important[/bold yellow]", style="yellow")
    console.print("• **Close all Chrome windows** using this profile")
    console.print("• Check Task Manager / Activity Monitor for chrome processes\n")

    if not Confirm.ask("Is Chrome fully closed?", default=True):
        console.print("[red]Exiting — close Chrome and retry.[/red]")
        return

    start_url = (
        Prompt.ask(
            "[bold cyan]Enter starting URL[/bold cyan]",
            default="https://www.google.com",
        ).strip()
        or "https://www.google.com"
    )

    console.print(f"\n[bold green]→ Target:[/bold green] {start_url}\n")

    # ── Launch browser ───────────────────────────────────────────────
    with sync_playwright() as p:
        try:
            console.print(
                "[green]Launching persistent Chrome (no stealth lib)...[/green]"
            )

            lock_file = os.path.join(user_data_dir, "SingletonLock")
            if os.path.exists(lock_file):
                console.print(
                    "[bold red]SingletonLock file still exists → Chrome is probably still running[/bold red]"
                )
                console.print("→ Close **all** Chrome windows and retry\n")
                time.sleep(2.5)

            context = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=False,
                channel="chrome",
                args=[
                    f"--profile-directory={profile_name}",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--window-size=1280,900",
                    "--start-maximized",
                    "--disable-dev-shm-usage",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                ],
                user_agent=user_agent,
                viewport={"width": 1280, "height": 900},
                locale="en-US",
                timezone_id="America/Los_Angeles",  # ← adjust to your timezone
                ignore_default_args=["--enable-automation"],
                bypass_csp=True,
                java_script_enabled=True,
            )

            # ── Reuse existing page if available (critical for persistent context) ──
            pages = context.pages
            if pages:
                page = pages[0]
                console.print(
                    "[dim]→ Reusing existing tab from persistent profile[/dim]"
                )
            else:
                page = context.new_page()
                console.print("[dim]→ Created new tab[/dim]")

            # Apply basic anti-detection script
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => false });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                window.chrome = { runtime: {}, app: {}, webstore: {} };
            """)

            # ── Navigation with progress ────────────────────────────────
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("[yellow]Loading...", total=100)

                progress.update(
                    task, description=f"[yellow]Navigating to {start_url}[/yellow]"
                )

                try:
                    response = page.goto(
                        start_url,
                        wait_until="domcontentloaded",  # more reliable than networkidle
                        timeout=60000,
                    )
                except TimeoutError:
                    console.print(
                        "[yellow]→ domcontentloaded timeout – continuing anyway[/yellow]"
                    )
                    response = None

                progress.update(task, advance=35)

                sleep(random.uniform(2.0, 4.5))

                # Additional safety waits
                try:
                    page.wait_for_selector("body", timeout=15000)
                    console.print("[dim]→ body element detected[/dim]")
                except:
                    console.print("[dim yellow]→ No body found after 15s[/dim yellow]")

                try:
                    page.wait_for_load_state("domcontentloaded", timeout=20000)
                except TimeoutError:
                    console.print(
                        "[dim]→ domcontentloaded state timeout – proceeding[/dim]"
                    )

                # Extra wait for JS-heavy sites
                with console.status(
                    "[dim]Waiting extra 5–9s for dynamic content…[/dim]",
                    spinner="dots",
                ):
                    sleep(random.uniform(5.0, 9.0))

                progress.update(task, advance=65)

                # ── Collect information right after navigation ───────
                title_immediate = page.title() or "(empty)"
                final_url = page.url
                webdriver_value = page.evaluate("navigator.webdriver")

                console.print(f"[dim]Current URL:[/dim] {final_url}")
                console.print(f"[dim]Title:[/dim] {title_immediate}")
                console.print(f"[dim]navigator.webdriver:[/dim] {webdriver_value}")

                progress.update(task, completed=100)

            # ── Prepare summary ──────────────────────────────────────
            status = response.status if response else "n/a"
            cookies_count = len(context.cookies())

            url_diff = list(
                difflib.unified_diff(
                    [start_url + "\n"],
                    [final_url + "\n"],
                    fromfile="start_url",
                    tofile="final_url",
                    lineterm="",
                )
            )

            meta = (
                f"[b]Title[/b]            : {title_immediate}\n"
                f"[b]Final URL[/b]        : {final_url}\n"
                f"[b]HTTP Status[/b]      : {status}\n"
                f"[b]navigator.webdriver[/b] : {webdriver_value}  [green](should be False)[/green]\n"
                f"[dim]Cookies count[/dim]    : {cookies_count}"
            )

            console.print(
                "\n"
                + Panel(
                    meta, title="Page Information", border_style="green", expand=False
                )
            )

            if any(line.startswith(("+", "-")) for line in url_diff[2:]):
                console.print("[bold cyan]URL changed during navigation[/bold cyan]")
                console.print(Markdown("```diff\n" + "\n".join(url_diff) + "\n```"))

            console.print(
                "\n[bold bright_green]Browser should now show the page[/bold bright_green]\n"
            )

            close_delay = Prompt.ask(
                "[cyan]Keep browser open for how many seconds?[/cyan]", default="15"
            )
            try:
                secs = int(close_delay)
            except:
                secs = 15

            with console.status(
                f"[dim]Keeping open for {secs} seconds (or close manually)...[/dim]",
                spinner="dots8Bit",
            ):
                page.wait_for_timeout(secs * 1000)

        except Exception as e:
            err_str = str(e).lower()
            if any(
                x in err_str
                for x in ["target closed", "existing", "already running", "lock"]
            ):
                console.print(
                    "\n[bold red]Chrome / profile appears to still be running![/bold red]\n"
                    "→ Close **every** Chrome window\n"
                    "→ Kill all chrome processes\n"
                    "→ Delete SingletonLock file if it remains"
                )
            else:
                console.print("\n[bold red]Playwright / launch error:[/bold red]")
                console.print(traceback.format_exc())
            raise

        finally:
            console.print("[dim]Closing context...[/dim]")
            time.sleep(3)
            try:
                context.close()
            except:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}", style="red")
        console.print(traceback.format_exc())
