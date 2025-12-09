import json
import httpx
import time
from pathlib import Path
from rich import print
from typing import Optional

BASE_URL = "http://shawn-pc.local:8001/transcribe_translate"

# Reusable global client (with pooling & HTTP/2 for local perf)
_client: Optional[httpx.Client] = None

def get_client() -> httpx.Client:
    global _client
    if _client is None:
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)  # Tune for local
        _client = httpx.Client(timeout=30.0, limits=limits, http2=True)  # http2 needs httpx[http2]
    return _client

def upload_file_multipart(file_path: Path) -> dict:
    client = get_client()
    with file_path.open("rb") as f:
        files = {"data": (file_path.name, f, "application/octet-stream")}
        print(f"[bold green]Sending as multipart:[/bold green] {file_path}")
        r = client.post(BASE_URL, files=files)
        return r.json()

def upload_raw_bytes(file_path: Path) -> dict:
    client = get_client()
    data = file_path.read_bytes()
    print(f"[bold yellow]Sending as raw bytes:[/bold yellow] {len(data):,} bytes")
    r = client.post(
        BASE_URL,
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    return r.json()

def main():
    file = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/data/sound.wav")

    start1 = time.perf_counter()
    result1 = upload_file_multipart(file)
    end1 = time.perf_counter()
    print(json.dumps(result1, indent=2, ensure_ascii=False))
    print(f"[bold green]upload_file_multipart duration:[/bold green] {end1 - start1:.3f} seconds")

    print("\n" + "─" * 50 + "\n")

    start2 = time.perf_counter()
    result2 = upload_raw_bytes(file)
    end2 = time.perf_counter()
    print(json.dumps(result2, indent=2, ensure_ascii=False))
    print(f"[bold yellow]upload_raw_bytes duration:[/bold yellow] {end2 - start2:.3f} seconds")

    # Cleanup on exit
    if _client:
        _client.close()

if __name__ == "__main__":
    main()