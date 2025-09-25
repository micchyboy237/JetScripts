from jet.transformers.formatters import format_json
from __future__ import annotations as _annotations
from dataclasses import dataclass
from deepeval.dataset import EvaluationDataset
from deepeval.integrations.pydantic_ai import Agent
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import BaseMetric
from httpx import AsyncClient
from jet.logger import logger
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from typing import Any
import asyncio
import os
import shutil

async def main():
    
    
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
    ## Evaluate Pydantic AI weather agent
    This tutorial will show you how to evaluate Pydantic AI agents using DeepEval's dataset iterator.
    
    ### Install dependencies:
    """
    logger.info("## Evaluate Pydantic AI weather agent")
    
    # !pip install pydantic-ai -U deepeval --quiet
    
    """
    ### Set your Ollama API key:
    """
    logger.info("### Set your Ollama API key:")
    
    
    # os.environ["OPENAI_API_KEY"] = "<your-ollama-api-key>"
    
    """
    ### Hyperparameters
    
    Hyperparameters of an LLM are the parameters that are used to control the behavior of the LLM application. It can be model, temperature, max tokens, or even you static prompts (for eg, system prompt). One of the main aim of performing evlauation is to find the best set of hyperparameters for a given agent.
    
    For this application, we are using model as one of the hyperparameter.
    """
    logger.info("### Hyperparameters")
    
    hyperparameter_model = "gpt-4o"
    
    """
    ### Create a Pydantic AI agent. 
    
    This is the same example as the one in the [Pydantic AI docs](https://ai.pydantic.dev/examples/weather-agent/). User can ask for the weather in multiple cities, the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use
    the `get_weather` tool to get the weather.
    """
    logger.info("### Create a Pydantic AI agent.")
    
    
    
    
    
    
    @dataclass
    class Deps:
        client: AsyncClient
    
    
    weather_agent = Agent(
        hyperparameter_model,
        instructions="Be concise, reply with one sentence.",
        deps_type=Deps,
        retries=2,
    )
    
    
    class LatLng(BaseModel):
        lat: float
        lng: float
    
    
    @weather_agent.tool
    async def get_lat_lng(
        ctx: RunContext[Deps], location_description: str
    ) -> LatLng:
        """Get the latitude and longitude of a location.
    
        Args:
            ctx: The context.
            location_description: A description of a location.
        """
        r = await ctx.deps.client.get(
                "https://demo-endpoints.pydantic.workers.dev/latlng",
                params={"location": location_description},
            )
        logger.success(format_json(r))
        r.raise_for_status()
        return LatLng.model_validate_json(r.content)
    
    
    @weather_agent.tool
    async def get_weather(
        ctx: RunContext[Deps], lat: float, lng: float
    ) -> dict[str, Any]:
        """Get the weather at a location.
    
        Args:
            ctx: The context.
            lat: Latitude of the location.
            lng: Longitude of the location.
        """
        temp_response, descr_response = await asyncio.gather(
                ctx.deps.client.get(
                    "https://demo-endpoints.pydantic.workers.dev/number",
                    params={"min": 10, "max": 30},
                ),
                ctx.deps.client.get(
                    "https://demo-endpoints.pydantic.workers.dev/weather",
                    params={"lat": lat, "lng": lng},
                ),
            )
        logger.success(format_json(temp_response, descr_response))
        temp_response.raise_for_status()
        descr_response.raise_for_status()
        return {
            "temperature": f"{temp_response.text} °C",
            "description": descr_response.text,
        }
    
    
    async def run_agent(input_query: str):
        async with AsyncClient() as client:
                deps = Deps(client=client)
                result = await weather_agent.run(input_query, deps=deps)
                return result.output
            
            
        logger.success(format_json(result))
    await run_agent(
        "What is the weather like in London and in Wiltshire?"
    )  # test run the agent
    
    """
    ### Evaluate the agent
    
    To evaluate Pydantic AI agents, use Deepeval's Pydantic AI `Agent` to supply metrics.
    
    
    > (Pro Tip) View your Agent's trace and publish test runs on [Confident AI](https://www.confident-ai.com/). Apart from this you get an in-house dataset editor and more advaced tools to monitor and enventually improve your Agent's performance. Get your API key from [here](https://app.confident-ai.com/)
    
    Given below is the code to instrument the application.
    """
    logger.info("### Evaluate the agent")
    
    
    instrument_pydantic_ai()
    
    """
    ### Dataset
    
    For evaluating the agent, we need a dataset. You can create your own dataset or use the one from the [Confident AI](https://www.confident-ai.com/docs/llm-evaluation/dataset-management/create-goldens).
    """
    logger.info("### Dataset")
    
    
    dataset = EvaluationDataset()
    dataset.pull(alias="weather_agent_queries", public=True)
    
    """
    ### Create a metric to evaluate the agent.
    
    Deepeval provides a state of the art ready to use [metric](https://deepeval.com/docs/metrics-introduction) to evaluate the agent. For this example, we will use the `AnswerRelevancyMetric`.
    
    > [!NOTE]
    You can only run end-to-end evals on metrics that evaluate the input and actual output of your Pydantic agent.
    
    Using Deepeval's Pydantic AI `Agent` wrapper, you can supply metrics to the agent.
    """
    logger.info("### Create a metric to evaluate the agent.")
    
    
    weather_agent = Agent(
        hyperparameter_model,
        instructions="Be concise, reply with one sentence.",
        deps_type=Deps,
        retries=2,
    )
    
    
    class LatLng(BaseModel):
        lat: float
        lng: float
    
    
    @weather_agent.tool
    async def get_lat_lng(
        ctx: RunContext[Deps], location_description: str
    ) -> LatLng:
        r = await ctx.deps.client.get(
                "https://demo-endpoints.pydantic.workers.dev/latlng",
                params={"location": location_description},
            )
        logger.success(format_json(r))
        r.raise_for_status()
        return LatLng.model_validate_json(r.content)
    
    
    @weather_agent.tool
    async def get_weather(
        ctx: RunContext[Deps], lat: float, lng: float
    ) -> dict[str, Any]:
    
        temp_response, descr_response = await asyncio.gather(
                ctx.deps.client.get(
                    "https://demo-endpoints.pydantic.workers.dev/number",
                    params={"min": 10, "max": 30},
                ),
                ctx.deps.client.get(
                    "https://demo-endpoints.pydantic.workers.dev/weather",
                    params={"lat": lat, "lng": lng},
                ),
            )
        logger.success(format_json(temp_response, descr_response))
        temp_response.raise_for_status()
        descr_response.raise_for_status()
        return {
            "temperature": f"{temp_response.text} °C",
            "description": descr_response.text,
        }
    
    
    async def run_agent(input_query: str, metrics: list[BaseMetric]):
        async with AsyncClient() as client:
                deps = Deps(client=client)
                result = await weather_agent.run(
                    input_query, deps=deps, metrics=metrics
                )
                return result.output
        logger.success(format_json(result))
    
    """
    ### Use the dataset iterator to evaluate the agent.
    
    Use the dataset iterator (from the dataset that was pulled earlier from the Confident AI) to evaluate the agent.
    """
    logger.info("### Use the dataset iterator to evaluate the agent.")
    
    
    for golden in dataset.evals_iterator():
        task = asyncio.create_task(
            run_agent(
                golden.input,
                metrics=[
                    AnswerRelevancyMetric(
                        threshold=0.7, model="llama3.2", include_reason=True
                    )
                ],
            )
        )
        dataset.evaluate(task)
    
    """
    Try changing hyperparameters and see how the agent performs.
    """
    logger.info("Try changing hyperparameters and see how the agent performs.")
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())