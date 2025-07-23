import asyncio
import os
from pathlib import Path
import shutil
import http.server
import socketserver
import webbrowser
import time

from jet.scrapers.automation.webpage_cloner import clone_after_render, generate_react_components, generate_entry_point


async def main():
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    # url = "http://example.com"
    url = "https://aniwatchtv.to"

    # Clone webpage
    await clone_after_render(url, output_dir, headless=False)

    # Generate React components
    html_path = Path(output_dir) / "index.html"
    html_content = html_path.read_text(encoding="utf-8")
    components = generate_react_components(html_content, output_dir)
    generate_entry_point(components, output_dir)
    print(f"Components generated in {output_dir}/components")
    print(f"Entry point generated at {output_dir}/index.html")

    # Start HTTP server
    port = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    os.chdir(output_dir)  # Change to output_dir to serve files
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving at http://localhost:{port} from {output_dir}")
        # Open the browser with a cache-busting query parameter for hard refresh
        webbrowser.open(f"http://localhost:{port}/?t={int(time.time())}")
        httpd.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
