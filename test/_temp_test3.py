import asyncio
import json
import httpx
from pathlib import Path
from rich import print

BASE_URL = "http://shawn-pc.local:8001"

async def upload_file_multipart(file_path: Path):
    async with httpx.AsyncClient(timeout=30.0) as client:
        with file_path.open("rb") as f:
            files = {"data": (file_path.name, f, "application/octet-stream")}
            print(f"[bold green]Sending as multipart:[/bold green] {file_path}")
            r = await client.post(f"{BASE_URL}/sample", files=files)
            print(json.dumps(r.json(), indent=2))

async def upload_raw_bytes(file_path: Path):
    async with httpx.AsyncClient(timeout=30.0) as client:
        data = file_path.read_bytes()
        print(f"[bold yellow]Sending as raw bytes:[/bold yellow] {len(data):,} bytes")
        r = await client.post(
            f"{BASE_URL}/sample",
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        print(json.dumps(r.json(), indent=2))

async def main():
    file = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/data/sound.wav")  # or any file: pdf, zip, model.bin, etc.

    await upload_file_multipart(file)
    print("\n" + "─" * 50 + "\n")
    await upload_raw_bytes(file)

if __name__ == "__main__":
    asyncio.run(main())