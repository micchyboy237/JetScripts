import requests
import json
import aiohttp
import asyncio
from jet.logger import logger
import uuid

BASE_URL = "http://localhost:9000"


def non_streaming_request():
    """Example of a non-streaming POST request to the /generate endpoint."""
    task_id = str(uuid.uuid4())
    payload = {
        "model": "llama-3.2-1b-instruct-4bit",
        "prompts": ["Write a short story.", "Explain AI in simple terms."],
        "max_tokens": 50,
        "temp": 0.8,
        "verbose": True,
        "task_id": task_id
    }
    try:
        response = requests.post(f"{BASE_URL}/generate", json=payload)
        response.raise_for_status()
        results = []
        for line in response.iter_lines(decode_unicode=True):
            if line:
                print(f"Received: {line}")
                if line.startswith("data: ") and "Result" in line:
                    result_json = json.loads(line.replace("data: ", ""))
                    results.append(result_json)
        print("\nNon-streaming results:")
        for result in results:
            print(f"Task ID: {result['task_id']}")
            print(f"Prompt ID: {result['prompt_id']}")
            print(f"Prompt: {result['prompt']}")
            print(f"Response: {result['response']}\n")
    except requests.exceptions.RequestException as e:
        print(f"Error in non-streaming request: {e}")


async def streaming_request():
    """Example of a streaming POST request to the /generate endpoint using aiohttp."""
    task_id = str(uuid.uuid4())
    payload = {
        "model": "llama-3.2-1b-instruct-4bit",
        "prompts": ["Tell a joke.", "What is machine learning?"],
        "max_tokens": 50,
        "temp": 0.7,
        "verbose": True,
        "task_id": task_id
    }
    headers = {"Accept": "application/json"}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{BASE_URL}/generate", json=payload, headers=headers) as response:
                response.raise_for_status()
                print("\nStreaming response:")
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line:
                            try:
                                data = json.loads(decoded_line)
                                print("\n--- New Chunk ---")
                                print(f"Type: {data['type']}")
                                if "task_id" in data:
                                    print(f"Task ID: {data['task_id']}")
                                if "prompt_id" in data:
                                    print(f"Prompt ID: {data['prompt_id']}")
                                if "prompt" in data:
                                    print(f"Prompt: {data['prompt']}")
                                if data['type'] == "token":
                                    print(f"Token: {data['token']}")
                                elif data['type'] == "result":
                                    print(f"Response: {data['response']}")
                                elif data['type'] == "error":
                                    print(f"Error Message: {data['message']}")
                                print("-----------------")
                            except json.JSONDecodeError:
                                logger.error(
                                    f"Invalid JSON in stream: {decoded_line}")
                                print(f"\n--- New Chunk ---")
                                print(f"Type: invalid")
                                print(f"Raw Data: {decoded_line}")
                                print("-----------------")
        except aiohttp.ClientError as e:
            print(f"Error in streaming request: {e}")

if __name__ == "__main__":
    print("\nRunning streaming example...")
    asyncio.run(streaming_request())

    # print("Running non-streaming example...")
    # non_streaming_request()
