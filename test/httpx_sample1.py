import asyncio
import httpx


async def call_streaming_endpoint():
    url = "http://127.0.0.1:8000/api/threads/test-thread/runs/stream"
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url) as response:
            if response.status_code == 200:
                async for line in response.aiter_lines():
                    print(line)  # Print each line as it arrives
            else:
                print(f"Error: {response.status_code}, {response.text}")

asyncio.run(call_streaming_endpoint())
