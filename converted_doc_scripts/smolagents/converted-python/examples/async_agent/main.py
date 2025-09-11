from jet.transformers.formatters import format_json
from jet.logger import logger
from smolagents import CodeAgent, InferenceClientModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
import anyio.to_thread
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
Async CodeAgent Example with Starlette

This example demonstrates how to use a CodeAgent in an async Starlette app,
running the agent in a background thread using anyio.to_thread.run_sync.
"""




# Create a simple agent instance (customize as needed)
def get_agent():
    # You can set custom model, or tools as needed
    return CodeAgent(
        model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
        tools=[],
    )


async def run_agent_in_thread(task: str):
    agent = get_agent()
    # The agent's run method is synchronous
    result = await anyio.to_thread.run_sync(agent.run, task)
    logger.success(format_json(result))
    return result


async def run_agent_endpoint(request: Request):
    data = await request.json()
    logger.success(format_json(data))
    task = data.get("task")
    if not task:
        return JSONResponse({"error": 'Missing "task" in request body.'}, status_code=400)
    try:
        result = await run_agent_in_thread(task)
        logger.success(format_json(result))
        return JSONResponse({"result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


routes = [
    Route("/run-agent", run_agent_endpoint, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)

logger.info("\n\n[DONE]", bright=True)