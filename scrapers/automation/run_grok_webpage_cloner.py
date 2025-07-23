import asyncio
import os
import shutil
import http.server
import socketserver
import webbrowser
from pathlib import Path

from jet.scrapers.automation.webpage_cloner import (
    clone_after_render,
    generate_react_components,
    generate_entry_point
)

SCRIPT_DIR = os.path.dirname(__file__)
PORT = 8000


async def run_pipeline() -> str:
    output_dir = os.path.join(
        SCRIPT_DIR,
        "generated",
        os.path.splitext(os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    # url = "http://example.com"
    # url = "https://www.iana.org/help/example-domains"
    # url = "https://www.w3schools.com/html/"
    # url = "https://aniwatchtv.to"
    url = "https://jethro-estrada.web.app"

    await clone_after_render(url, output_dir, headless=False)

    html_path = Path(output_dir) / "index.html"
    html_content = html_path.read_text(encoding="utf-8")
    components = generate_react_components(html_content, output_dir)
    generate_entry_point(components, output_dir)

    print(f"Components generated in {output_dir}/components")
    print(f"Entry point generated at {output_dir}/index.html")

    return output_dir


async def serve_once(directory: str):
    Handler = http.server.SimpleHTTPRequestHandler
    os.chdir(directory)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/"
        print(f"Serving at {url} from {directory}")
        webbrowser.open(url)  # Reuse existing tab if open
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.")


async def main():
    try:
        output_dir = await run_pipeline()
        await serve_once(output_dir)
    except Exception as e:
        print(f"\n⚠️ Error occurred: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
