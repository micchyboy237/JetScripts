import asyncio
from fastapi import Depends, HTTPException, Header
from fastapi import FastAPI
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from jet.logger import CustomLogger
from pydantic import BaseModel
from scripts.fastapi_agent import app
from sse_starlette.sse import EventSourceResponse
from typing import Optional
import asyncio
import json
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--fastapi-agent--fastapi-agent-tutorial)

# Serving an Agent with FastAPI

## What is FastAPI and Why Use It for Agents?

FastAPI is a modern, high-performance web framework for building APIs with Python. Released in 2018, it has quickly gained popularity due to its combination of speed, ease of use, and developer-friendly features.

At its core, FastAPI is designed to create REST APIs that can serve requests efficiently while providing robust validation and documentation. For AI agent deployment, FastAPI offers several critical advantages:

- **Asynchronous Support**: AI agents often need to handle concurrent requests efficiently. FastAPI's native async/await support enables handling thousands of simultaneous connections, perfect for serving multiple agent requests in parallel without blocking.

- **Streaming Responses**: Agents frequently generate content incrementally (token by token). FastAPI's streaming response capabilities allow for real-time transmission of agent outputs as they're generated, creating a more responsive user experience.

- **Type Validation**: When working with agents, ensuring proper input formats is crucial. FastAPI uses Pydantic for automatic request validation, catching malformed inputs before they reach your agent and providing clear error messages.

- **Performance**: Built on Starlette and Uvicorn, FastAPI offers near-native performance. For compute-intensive agent applications, this means your infrastructure handles API overhead efficiently, allowing more resources for the actual agent processing.

- **Automatic Documentation**: When exposing an agent API to multiple users or teams, documentation becomes essential. FastAPI automatically generates interactive API documentation via Swagger UI and ReDoc, making it easy for others to understand and use your agent.

- **Schema Enforcement**: Pydantic models ensure that both requests to your agent and responses from it conform to predefined schemas, making agent behavior more predictable and easier to integrate with other systems.

In this tutorial, we'll build a complete API that serves an AI agent with both synchronous and streaming endpoints, demonstrating how FastAPI's features address the specific challenges of deploying agents in production.

## Prerequisites

Before we begin, let's install the necessary packages:
"""
logger.info("# Serving an Agent with FastAPI")

# !pip install fastapi uvicorn pydantic

"""
If you plan to use the streaming functionality, also install:
"""
logger.info("If you plan to use the streaming functionality, also install:")

# !pip install sse-starlette

"""
## Agent Quick Recap

Let's start by defining a simple agent that we'll expose via our API. This could be any agent implementation, but for this tutorial, we'll create a basic example that simulates an AI agent responding to user queries:
"""
logger.info("## Agent Quick Recap")

class SimpleAgent:
    def __init__(self, name="FastAPI Agent"):
        self.name = name

    def generate_response(self, query):
        """Generate a synchronous response to a user query"""
        return f"Agent {self.name} received: '{query}'\nResponse: This is a simulated agent response."

    async def generate_response_stream(self, query):
        """Generate a streaming response to a user query"""

        prefix = f"Agent {self.name} thinking about: '{query}'\n"
        response = "This is a simulated agent response that streams token by token."

        yield prefix

        for token in response.split():
            async def run_async_code_69016fd9():
                await asyncio.sleep(0.1)  # Simulate thinking time
            asyncio.run(run_async_code_69016fd9())
            yield token + " "

agent = SimpleAgent()
test_query = "Hello, what can you do?"
logger.debug(agent.generate_response(test_query))

"""
This simple agent can generate both synchronous responses and streaming responses. In practice, you might replace this with a more sophisticated agent like a fine-tuned LLM, an RAG system, or any other AI agent.

## Minimal FastAPI App

Now, let's create a minimal FastAPI application with a health check endpoint:
"""
logger.info("## Minimal FastAPI App")


app = FastAPI(
    title="Agent API",
    description="A simple API that serves an AI agent",
    version="0.1.0"
)

agent = SimpleAgent()

@app.get("/health")
def health_check():
    """Check if the API is running"""
    return {"status": "ok", "message": "API is operational"}

"""
This creates a basic FastAPI application with metadata and a health check endpoint. The health check is a simple way to verify that your API is running correctly.

## POST /agent - Synchronous Endpoint

Now, let's create a synchronous endpoint for our agent. We'll use Pydantic models to define the request and response structures:
"""
logger.info("## POST /agent - Synchronous Endpoint")


class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "query": "What is FastAPI?",
                "context": "I'm a beginner programmer."
            }
        }

class QueryResponse(BaseModel):
    response: str

    class Config:
        schema_extra = {
            "example": {
                "response": "FastAPI is a modern, high-performance web framework for building APIs with Python."
            }
        }

@app.post("/agent", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    """Get a response from the agent"""
    response = agent.generate_response(request.query)
    return QueryResponse(response=response)

"""
This endpoint accepts POST requests with a JSON body containing a "query" field and an optional "context" field. It returns a JSON response with the agent's answer.

## POST /agent/stream - Token Streaming

For many AI applications, token streaming provides a better user experience. Let's implement a streaming endpoint:
"""
logger.info("## POST /agent/stream - Token Streaming")


@app.post("/agent/stream")
async def stream_agent(request: QueryRequest):
    """Stream a response from the agent token by token"""

    async def event_generator():
        async for token in agent.generate_response_stream(request.query):
            data = json.dumps({"token": token})
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

"""
This endpoint streams the agent's response token by token using Server-Sent Events (SSE). The client can process these tokens incrementally as they arrive, enabling a more interactive experience.

For a more sophisticated implementation, you might want to use the `sse-starlette` package:
"""
logger.info("This endpoint streams the agent's response token by token using Server-Sent Events (SSE). The client can process these tokens incrementally as they arrive, enabling a more interactive experience.")


@app.post("/agent/stream-sse")
async def stream_agent_sse(request: QueryRequest):
    """Stream a response using SSE with the sse-starlette package"""

    async def event_generator():
        async for token in agent.generate_response_stream(request.query):
            yield {"data": json.dumps({"token": token})}

    return EventSourceResponse(event_generator())

"""
This provides a more robust implementation of Server-Sent Events.

## Creating the Full Application

Now, let's put everything together into a complete FastAPI application. Create a file named `fastapi_agent.py` in your `scripts` directory:
"""
logger.info("## Creating the Full Application")


class SimpleAgent:
    def __init__(self, name="FastAPI Agent"):
        self.name = name

    def generate_response(self, query):
        """Generate a synchronous response to a user query"""
        return f"Agent {self.name} received: '{query}'\nResponse: This is a simulated agent response."

    async def generate_response_stream(self, query):
        """Generate a streaming response to a user query"""
        prefix = f"Agent {self.name} thinking about: '{query}'\n"
        response = "This is a simulated agent response that streams token by token."

        yield prefix

        for token in response.split():
            async def run_async_code_69016fd9():
                await asyncio.sleep(0.1)  # Simulate thinking time
            asyncio.run(run_async_code_69016fd9())
            yield token + " "

class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "query": "What is FastAPI?",
                "context": "I'm a beginner programmer."
            }
        }

class QueryResponse(BaseModel):
    response: str

    class Config:
        schema_extra = {
            "example": {
                "response": "FastAPI is a modern, high-performance web framework for building APIs with Python."
            }
        }

app = FastAPI(
    title="Agent API",
    description="A simple API that serves an AI agent",
    version="0.1.0"
)

agent = SimpleAgent()

@app.get("/health")
def health_check():
    """Check if the API is running"""
    return {"status": "ok", "message": "API is operational"}

@app.post("/agent", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    """Get a response from the agent"""
    response = agent.generate_response(request.query)
    return QueryResponse(response=response)

@app.post("/agent/stream")
async def stream_agent(request: QueryRequest):
    """Stream a response from the agent token by token"""

    async def event_generator():
        async for token in agent.generate_response_stream(request.query):
            data = json.dumps({"token": token})
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

"""
## Running the Server

Now that we have our FastAPI application, let's run it with uvicorn:
"""
logger.info("## Running the Server")

# !cd tutorials/fastapi-agent && uvicorn fastapi_agent:app --reload

"""
The `--reload` flag enables hot reloading, which automatically restarts the server when you make changes to the code. This is helpful during development.

Once running, you can access:
- API documentation at http://localhost:8000/docs
- Alternative documentation at http://localhost:8000/redoc
- Health check endpoint at http://localhost:8000/health

## Simple Client Test

Let's test our API with a simple Python client:
"""
logger.info("## Simple Client Test")


response = requests.post(
    "http://localhost:8000/agent",
    json={"query": "What is FastAPI?"}
)
logger.debug("Synchronous Response:")
logger.debug(response.json())
logger.debug("\n" + "-" * 40 + "\n")

response = requests.post(
    "http://localhost:8000/agent/stream",
    json={"query": "Tell me about streaming"},
    stream=True
)

logger.debug("Streaming Response:")
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            logger.debug(data["token"], end="")

"""
This script tests both the synchronous and streaming endpoints of our API.

## Adding Basic Auth Key (Optional)

For production use, you might want to add simple API key authentication. Let's extend our FastAPI application to check for an API key:
"""
logger.info("## Adding Basic Auth Key (Optional)")


async def verify_api_key(x_api_key: str = Header(None)):
    """Verify the API key provided in the X-API-Key header"""
    api_key = os.environ.get("API_KEY")

    if not api_key:
        return True

    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is missing")

    if x_api_key != api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    return True

@app.post("/agent", response_model=QueryResponse)
def query_agent(request: QueryRequest, authorized: bool = Depends(verify_api_key)):
    """Get a response from the agent"""
    response = agent.generate_response(request.query)
    return QueryResponse(response=response)

@app.post("/agent/stream")
async def stream_agent(request: QueryRequest, authorized: bool = Depends(verify_api_key)):
    """Stream a response from the agent token by token"""

"""
With this update, if you set the `API_KEY` environment variable, the API will require a matching key in the `X-API-Key` header for all requests.

## Unit Tests

Let's create simple unit tests for our FastAPI application using pytest and the FastAPI test client:
"""
logger.info("## Unit Tests")


client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_agent_endpoint():
    """Test the synchronous agent endpoint"""
    response = client.post(
        "/agent",
        json={"query": "Test query"}
    )
    assert response.status_code == 200
    assert "response" in response.json()
    assert "Agent" in response.json()["response"]

def test_stream_endpoint():
    """Test the streaming agent endpoint"""
    with client.stream("POST", "/agent/stream", json={"query": "Test query"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        content = response.iter_content().read()
        assert len(content) > 0

"""
Save these tests in a file named `test_fastapi_agent.py` in your tests directory and run them with pytest:
"""
logger.info("Save these tests in a file named `test_fastapi_agent.py` in your tests directory and run them with pytest:")

# !pytest -xvs tests/test_fastapi_agent.py

"""
## Next Steps

Now that you have a basic FastAPI agent service running, here are some ideas for next steps:

- **Add more advanced agents**: Replace the simple agent with your production-ready agent
- **Implement authentication and rate limiting**: Add more sophisticated authentication and rate limiting for production use
- **Add middleware for logging and monitoring**: Implement middleware for request logging and performance monitoring
- **Set up deployment**: Deploy your FastAPI application to a production environment using Docker, Kubernetes, or a cloud service
- **Implement async database connections**: Add database integrations for storing conversation history or other data
- **Add background tasks**: Use FastAPI's background tasks for long-running operations

## Conclusion

In this tutorial, we've built a FastAPI application that serves a simple AI agent with both synchronous and streaming endpoints. We've covered the basics of setting up FastAPI, defining Pydantic models for request/response validation, implementing both synchronous and streaming endpoints, and adding simple authentication.

FastAPI's combination of performance, automatic documentation, and developer-friendly features makes it an excellent choice for serving AI agents in production. By following the patterns in this tutorial, you can create robust, production-ready APIs for your own AI agents.
"""
logger.info("## Next Steps")

logger.info("\n\n[DONE]", bright=True)