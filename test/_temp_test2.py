from urllib.parse import urljoin

from openai import OpenAI
from trafilatura import extract, fetch_url

QUERY = "chunking strategies for PDF tables"
ROOT_URL = "https://docs.unstructured.io"
MAX_PAGES = 10

client = OpenAI(api_key="YOUR_API_KEY")


def is_relevant(text: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Does this text discuss '{QUERY}'? Reply JSON: {{'relevant': bool, 'summary': str}}",
            }
        ],
        response_format={"type": "json_object"},
    )
    return eval(resp.choices[0].message.content)


def trafilatura_crawl():
    collected, visited, queue = [], set(), [ROOT_URL]

    while queue and len(collected) < 3 and len(visited) < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        downloaded = fetch_url(url)
        if not downloaded:
            continue

        text = extract(downloaded, include_tables=True, output_format="markdown")
        if not text or len(text) < 200:
            continue

        check = is_relevant(text)
        if check["relevant"]:
            collected.append(
                {"url": url, "content": text[:3000], "summary": check["summary"]}
            )
            print(f"✅ RELEVANT [{len(collected)}]: {url}")
        else:
            print(f"❌ SKIP: {url}")

        # MANUAL link discovery (trafilatura doesn't do this)
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(downloaded, "html.parser")
        for a in soup.find_all("a", href=True):
            full = urljoin(url, a["href"])
            if full.startswith(ROOT_URL) and full not in visited:
                queue.append(full)

    print(f"\n=== FINAL CONTEXT ({len(collected)} pages) ===")
    for c in collected:
        print(f"\n📄 {c['url']}\n{c['summary']}")


trafilatura_crawl()
