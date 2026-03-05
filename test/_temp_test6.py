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

    profile_name = "Default"  # ← change if needed
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

    # ── Show config ──────────────────────────────────────────────────
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
                "[green]Launching persistent Chrome (without stealth lib)...[/green]"
            )

            lock_file = os.path.join(user_data_dir, "SingletonLock")
            if os.path.exists(lock_file):
                console.print(
                    "[bold red]SingletonLock still exists → Chrome/profile is probably still running[/bold red]"
                )
                console.print("→ Close all Chrome windows and retry\n")
                time.sleep(2.5)

            context = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=False,
                channel="chrome",  # use real installed Chrome
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
                    "--disable-features=site-per-process",  # sometimes helps
                ],
                user_agent=user_agent,
                viewport={"width": 1280, "height": 900},
                locale="en-US",
                timezone_id="America/Los_Angeles",  # ← change if needed
                ignore_default_args=["--enable-automation"],
                java_script_enabled=True,
                bypass_csp=True,
            )

            page = context.new_page()

            # Some extra common evasion tricks (without stealth lib)
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => false });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5].map(() => ({ length: 1 }))
                });
                window.chrome = { runtime: {}, app: {}, webstore: {} };
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
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
                response = page.goto(start_url, wait_until="networkidle", timeout=60000)
                progress.update(task, advance=40)

                sleep(random.uniform(1.8, 4.2))

                try:
                    page.wait_for_load_state("networkidle", timeout=18000)
                except TimeoutError:
                    console.print(
                        "[dim]→ networkidle timeout — continuing anyway[/dim]"
                    )

                # Many SPAs / anti-bot pages keep working after networkidle
                with console.status(
                    "[dim]Waiting extra time for JS & anti-bot checks…[/dim]",
                    spinner="dots",
                ):
                    sleep(random.uniform(5.5, 10.0))

                progress.update(task, advance=60)

                # ── Collect info ────────────────────────────────────────
                title_immediate = page.title()
                try:
                    page.wait_for_function("document.title.trim() !== ''", timeout=7000)
                except:
                    pass

                title_final = page.title()
                final_url = page.url

                progress.update(task, completed=100)

            # ── Show results ────────────────────────────────────────────
            status = response.status if response else "n/a"
            webdriver = page.evaluate("navigator.webdriver")
            cookies_count = len(context.cookies())

            url_diff = list(
                difflib.unified_diff(
                    [start_url + "\n"],
                    [final_url + "\n"],
                    fromfile="start_url (typed)",
                    tofile="final_url (after navigation)",
                    lineterm="",
                )
            )

            title_diff = list(
                difflib.unified_diff(
                    [title_immediate + "\n"],
                    [title_final + "\n"],
                    fromfile="title (immediate)",
                    tofile="title (after wait)",
                    lineterm="",
                )
            )

            meta = (
                f"[b]Immediate title[/b] : {title_immediate}\n"
                f"[b]Final title[/b]     : {title_final}\n"
                f"[b]Final URL[/b]       : {final_url}\n"
                f"[b]HTTP Status[/b]     : {status}\n"
                f"[b]navigator.webdriver[/b] : {webdriver}  [green](should be False!)[/green]\n"
                f"[dim]Cookies count[/dim]   : {cookies_count}"
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

            if title_immediate.strip() != title_final.strip():
                console.print("[bold cyan]Title changed after extra wait[/bold cyan]")
                console.print(Markdown("```diff\n" + "\n".join(title_diff) + "\n```"))

            console.print(
                "\n[bold bright_green]Browser launched successfully (no stealth lib)[/bold bright_green]\n"
            )

            close_delay = Prompt.ask(
                "[cyan]Keep browser open for how many seconds?[/cyan]", default="12"
            )
            try:
                secs = int(close_delay)
            except:
                secs = 12

            with console.status(
                f"[dim]Keeping open for {secs} seconds (or close manually)...[/dim]",
                spinner="dots8Bit",
            ):
                page.wait_for_timeout(secs * 1000)

        except Exception as e:
            err = str(e).lower()
            if any(
                x in err
                for x in [
                    "target closed",
                    "existing browser session",
                    "already running",
                ]
            ):
                console.print(
                    "\n[bold red]Chrome is still running with this profile![/bold red]\n"
                    "→ Close **every** Chrome window\n"
                    "→ Kill all chrome processes in Task Manager / Activity Monitor"
                )
            else:
                console.print("\n[bold red]Launch / Playwright error:[/bold red]")
                console.print(traceback.format_exc())
            raise

        finally:
            console.print("[dim]Closing context...[/dim]")
            sleep(3)
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
