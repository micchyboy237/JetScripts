from typing import Any
from pydantic import BaseModel
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Assuming Deps is a dataclass for dependencies (e.g., for tools); adjust as needed.
from dataclasses import dataclass
@dataclass
class Deps:
    some_dep: Any = None  # Placeholder; replace with your actual deps.

# Custom OpenAI client for local llama.cpp server.
client = AsyncOpenAI(base_url="http://shawn-pc.local:8080/v1")  # No api_key needed for local.

# Provider with the custom client.
provider = OpenAIProvider(openai_client=client)

# Model configured for your local setup.
model = OpenAIChatModel(model_name='gpt-5', provider=provider)  # Use your llama.cpp model name.

# Agent with deps_type.
agent = Agent(model, deps_type=Deps)

# Example run (sync for simplicity; use run() for async).
result = agent.run_sync("Explain quantum computing in one sentence.")
print(result.output)