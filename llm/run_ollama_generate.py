import os
import json
from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any
import aiohttp
from jet.logger import logger


class ChatLLM:
    class Config(BaseModel):
        api_url: str = Field(default="http://jetairm1:11434",
                             description="OpenAI compatible endpoint")
        model: str = Field(default="llama3.1:latest",
                           description="Model version to use")

    def __init__(self):
        self.config = self.Config()

    async def query_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Query the LLM with user input and optional system prompt."""

        url = f"{self.config.api_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True
        }

        logger.log("PAYLOAD:")
        logger.debug(json.dumps(payload, indent=2))

        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                return content
        except aiohttp.ClientError as e:
            raise Exception(f"Error querying LLM API: {e}")

    async def handle_message(self, message: str, event_emitter: Callable[[Any], Awaitable[None]]) -> None:
        """Handle incoming messages."""
        response = await self.query_llm("General system prompt", message)
        await event_emitter({"type": "message", "data": {"response": response}})


def load_prompt(prompt_path):
    # Check if the file exists
    if not os.path.exists(prompt_path):
        print(f"Error: The file at {prompt_path} is missing or not synced.")
        return None

    # Open and read the contents of the file
    with open(prompt_path, 'r') as file:
        return file.read()


if __name__ == "__main__":
    prompt_path = "/Users/jethroestrada/Library/Mobile Documents/com~apple~CloudDocs/Jet iCloud Files/Scripts/docs/llm_eval/React_JSON_Prompt.md"
    prompt = load_prompt(prompt_path)

    if prompt:
        print("Prompt loaded successfully.")
    else:
        print("Failed to load the prompt.")
