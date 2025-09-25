"""Usage examples for TavilySearchResults and TavilyAnswer tools.

Setup Instructions:
1. Install dependencies:
   ```bash
   pip install -U langchain-community tavily-python pytest pytest-asyncio redis
   ```
2. Set environment variable:
   ```bash
   export TAVILY_API_KEY="your-api-key"
   ```
3. Start Redis server:
   ```bash
   brew install redis
   redis-server --port 3103
   ```

These examples demonstrate synchronous and asynchronous searches, direct answer retrieval,
and tool call handling with type safety and caching.
"""

import asyncio
from typing import Dict, List, Literal, Optional, Tuple, TypedDict
from jet.search.tavily import TavilySearchResults, TavilyAnswer
from langchain_core.messages import ToolMessage

# Typed dictionaries for structured inputs and outputs


class SearchConfig(TypedDict):
    """Configuration for Tavily search."""
    max_results: int
    search_depth: Literal["basic", "advanced"]
    include_answer: bool
    include_raw_content: bool
    include_images: bool
    redis_host: str
    redis_port: int


class SearchResult(TypedDict):
    """Structure for a single search result."""
    title: str
    url: str
    content: str
    score: Optional[float]
    raw_content: Optional[str]


class SearchArtifact(TypedDict):
    """Structure for the full search artifact."""
    query: str
    results: List[SearchResult]
    answer: Optional[str]
    images: Optional[List[str]]
    response_time: Optional[float]
    follow_up_questions: Optional[List[str]]


class ToolCallInput(TypedDict):
    """Structure for tool call input."""
    args: Dict[str, str]
    type: Literal["tool_call"]
    id: str
    name: str


def example_sync_search() -> None:
    """Demonstrate synchronous search with TavilySearchResults."""
    # Define configuration
    config: SearchConfig = {
        "max_results": 3,
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": True,
        "include_images": True,
        "redis_host": "localhost",
        "redis_port": 3103
    }

    # Instantiate tool
    tool = TavilySearchResults(
        max_results=config["max_results"],
        search_depth=config["search_depth"],
        include_answer=config["include_answer"],
        include_raw_content=config["include_raw_content"],
        include_images=config["include_images"],
        redis_config={"host": config["redis_host"],
                      "port": config["redis_port"]}
    )

    # Perform search
    query = "Who won the last French Open?"
    result, artifact = tool.invoke({"query": query})

    # Print results
    print("Synchronous Search Results:")
    for item in result:
        print(f"- {item['title']}: {item['url']}")
    print("\nArtifact:")
    print(f"Answer: {artifact.get('answer', 'N/A')}")
    print(f"Images: {artifact.get('images', [])}")


async def example_async_search() -> None:
    """Demonstrate asynchronous search with TavilySearchResults."""
    # Define configuration
    config: SearchConfig = {
        "max_results": 2,
        "search_depth": "basic",
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
        "redis_host": "localhost",
        "redis_port": 3103
    }

    # Instantiate tool
    tool = TavilySearchResults(
        max_results=config["max_results"],
        search_depth=config["search_depth"],
        include_answer=config["include_answer"],
        include_raw_content=config["include_raw_content"],
        include_images=config["include_images"],
        redis_config={"host": config["redis_host"],
                      "port": config["redis_port"]}
    )

    # Perform async search
    query = "Latest AI advancements 2025"
    result, artifact = await tool.ainvoke({"query": query})

    # Print results
    print("Asynchronous Search Results:")
    for item in result:
        print(f"- {item['title']}: {item['url']}")
    print("\nArtifact:")
    print(f"Answer: {artifact.get('answer', 'N/A')}")


def example_get_answer() -> None:
    """Demonstrate retrieving a direct answer with TavilyAnswer."""
    # Instantiate tool
    tool = TavilyAnswer()

    # Get answer
    query = "What is the capital of France?"
    answer = tool.invoke({"query": query})

    # Print answer
    print(f"Answer: {answer}")


def example_tool_call() -> None:
    """Demonstrate handling a tool call with TavilySearchResults."""
    # Define configuration
    config: SearchConfig = {
        "max_results": 4,
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": False,
        "include_images": True,
        "redis_host": "localhost",
        "redis_port": 3103
    }

    # Instantiate tool
    tool = TavilySearchResults(
        max_results=config["max_results"],
        search_depth=config["search_depth"],
        include_answer=config["include_answer"],
        include_raw_content=config["include_raw_content"],
        include_images=config["include_images"],
        redis_config={"host": config["redis_host"],
                      "port": config["redis_port"]}
    )

    # Define tool call
    tool_call: ToolCallInput = {
        "args": {"query": "Recent Mars mission updates"},
        "type": "tool_call",
        "id": "call_123",
        "name": "tavily_search_results_json"
    }

    # Handle tool call
    result = tool.invoke(tool_call)

    # Print results
    print(f"Tool Message Content: {result.content}")
    print(f"Artifact: {result.artifact}")


def main() -> None:
    """Run all usage examples."""
    print("Running Example 1: Synchronous Search")
    example_sync_search()
    print("\nRunning Example 2: Asynchronous Search")
    asyncio.run(example_async_search())
    print("\nRunning Example 3: Get Direct Answer")
    example_get_answer()
    print("\nRunning Example 4: Tool Call")
    example_tool_call()


if __name__ == "__main__":
    main()
