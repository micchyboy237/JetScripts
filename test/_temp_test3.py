from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from defuddle import Defuddle
from openai import OpenAI

QUERY = "chunking strategies for PDF tables"
ROOT_URL = "https://docs.unstructured.io"

client = OpenAI(api_key="YOUR_API_KEY")


def defuddle_crawl():
    collected, visited, queue = [], set(), [ROOT_URL]

    while queue and len(collected) < 3 and len(visited) < 10:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        html = requests.get(url).text
        parsed = Defuddle(html).parse()
        md = parsed.content

        if not md or len(md) < 200:
            continue

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Relevant to '{QUERY}'? JSON: {{'relevant':bool,'summary':str}}",
                }
            ],
            response_format={"type": "json_object"},
        )
        check = eval(resp.choices[0].message.content)

        if check["relevant"]:
            collected.append(
                {"url": url, "markdown": md[:3000], "summary": check["summary"]}
            )
            print(f"✅ [{len(collected)}] {url}")

        # Again, FULLY MANUAL link discovery
        for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
            full = urljoin(url, a["href"])
            if full.startswith(ROOT_URL) and full not in visited:
                queue.append(full)

    for c in collected:
        print(f"\n📄 {c['url']}: {c['summary']}")


defuddle_crawl()
