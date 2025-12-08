import httpx

text = "今日はいい天気ですね。一一起に散歩しませんか？"
text = [
    "昨日、友達と一緒に映画を見に行きました。",
    "日本は美しい国ですね！",
    "今日の天気はとても良いです。",
]

client = httpx.Client(follow_redirects=True)   # ← this swallows 307 automatically
if isinstance(text, str):
    text = [text]
resp = client.post(
    "http://shawn-pc.local:8001/translate",         # no slash needed here
    json={"text": text},
    params={"device": "cuda"},
)
resp.raise_for_status()
response = resp.json()
results = response["results"]

for result in results:
    print("Original:   ", result["original"])
    print("Translation:", result["translation"])
client.close()