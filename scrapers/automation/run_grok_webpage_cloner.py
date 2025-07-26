import asyncio
import os
import shutil
import http.server
import socketserver
import webbrowser
import sys
import re
import string
import socket
from pathlib import Path
from urllib.parse import urlparse

from jet.scrapers.automation.webpage_cloner import (
    clone_after_render,
    generate_react_components,
    generate_entry_point
)

SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_PORT = 8000
PORT_RANGE = range(8000, 8100)  # Range of ports to try


def find_available_port(start_port: int, port_range: range) -> int:
    for port in port_range:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available ports found in the specified range")


def format_sub_url_dir(url: str) -> str:
    clean_url = re.sub(r'^(https?://|www\.)|(\?.*)', '', url)
    trans_table = str.maketrans({p: '_' for p in string.punctuation})
    formatted = clean_url.translate(trans_table).lower()
    formatted = re.sub(r'_+', '_', formatted)
    return formatted.strip('_')


async def run_pipeline(url) -> str:
    output_dir = os.path.join(
        SCRIPT_DIR,
        "generated",
        os.path.splitext(os.path.basename(__file__))[0]
    )

    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

    sub_url_dir = format_sub_url_dir(url)
    sub_output_dir = os.path.join(output_dir, sub_url_dir)

    shutil.rmtree(sub_output_dir, ignore_errors=True)

    await clone_after_render(url, sub_output_dir, headless=False)

    html_path = Path(sub_output_dir) / "index.html"
    html_content = html_path.read_text(encoding="utf-8")
    components, font_urls = generate_react_components(
        html_content, sub_output_dir, base_url=url)
    generate_entry_point(components, sub_output_dir, font_urls)

    print(f"Components generated in {sub_output_dir}/components")
    print(f"Entry point generated at {sub_output_dir}/index.html")

    return sub_output_dir


async def serve_once(directory: str):
    Handler = http.server.SimpleHTTPRequestHandler
    os.chdir(directory)

    port = find_available_port(DEFAULT_PORT, PORT_RANGE)
    with socketserver.TCPServer(("", port), Handler) as httpd:
        url = f"http://localhost:{port}/"
        print(f"Serving at {url} from {directory}")
        webbrowser.open(url)  # Reuse existing tab if open
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.")


async def main(url):
    try:
        output_dir = await run_pipeline(url)
        await serve_once(output_dir)
    except Exception as e:
        print(f"\n⚠️ Error occurred: {e}\n")

if __name__ == "__main__":
    # url = "http://example.com"
    url = "https://www.iana.org/help/example-domains"
    # url = "https://www.w3schools.com/html/"
    # url = "https://aniwatchtv.to"
    # url = "https://www.meetyourproperty.ph"
    # url = "https://jethro-estrada.web.app"

    if len(sys.argv) > 1:
        url = sys.argv[1]

    asyncio.run(main(url))
