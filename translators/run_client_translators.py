import httpx

client = httpx.Client(follow_redirects=True)   # ← this swallows 307 automatically
resp = client.post(
    "http://shawn-pc.local:8001/translate",         # no slash needed here
    json={"text": "今日はいい天気ですね。一一起に散歩しませんか？"},
    params={"device": "cuda"},
)
resp.raise_for_status()
result = resp.json()
print("Original:   ", result["original"])
print("Translation:", result["translation"])
client.close()