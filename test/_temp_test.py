import requests
import json
import aiohttp
import asyncio
from jet.logger import logger

# Base URL for the API
BASE_URL = "http://localhost:9000"


def non_streaming_request():
    """Example of a non-streaming POST request to the /generate endpoint."""
    payload = {
        "model": "llama-3.2-1b-instruct-4bit",
        "prompts": ["Write a short story.", "Explain AI in simple terms."],
        "max_tokens": 50,
        "temp": 0.8,
        "verbose": True
    }

    try:
        response = requests.post(f"{BASE_URL}/generate", json=payload)
        response.raise_for_status()

        # Collect all results
        results = []
        for line in response.iter_lines(decode_unicode=True):
            if line:
                print(f"Received: {line}")
                if line.startswith("result: "):
                    result_json = json.loads(
                        line.replace("result: Result ", "")[2:])
                    results.append(result_json)

        print("\nNon-streaming results:")
        for result in results:
            print(f"Prompt: {result['prompt']}")
            print(f"Response: {result['response']}\n")

    except requests.exceptions.RequestException as e:
        print(f"Error in non-streaming request: {e}")


async def streaming_request():
    """Example of a streaming POST request to the /generate endpoint using aiohttp."""
    payload = {
        "model": "llama-3.2-1b-instruct-4bit",
        "prompts": ["Tell a joke.", "What is machine learning?"],
        "max_tokens": 50,
        "temp": 0.7,
        "verbose": True
    }

    headers = {"Accept": "text/event-stream"}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{BASE_URL}/generate", json=payload, headers=headers) as response:
                response.raise_for_status()
                print("\nStreaming response:")

                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line:  # Only process non-empty lines
                            logger.success(decoded_line, flush=True)

        except aiohttp.ClientError as e:
            print(f"Error in streaming request: {e}")

if __name__ == "__main__":
    # Run non-streaming example
    print("Running non-streaming example...")
    non_streaming_request()

    # Run streaming example
    # print("\nRunning streaming example...")
    # asyncio.run(streaming_request())
