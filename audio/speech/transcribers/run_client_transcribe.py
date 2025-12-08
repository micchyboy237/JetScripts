import httpx

audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/data/sound.wav"

client = httpx.Client(follow_redirects=True)   # ← this swallows 307 automatically
if isinstance(text, str):
    text = [text]
resp = client.post(
    "http://shawn-pc.local:8001/transcribe",         # no slash needed here
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