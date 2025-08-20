import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import (
Context,
Event,
StartEvent,
StopEvent,
Workflow,
step,
)
from llama_index.llms.groq import Groq
from toolhouse import Toolhouse, Provider
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/gist/iamdaniele/84cca60019384c4159df28e14e2dc61c/toolhouse-llamaindex-workflow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Sales Prospecting Workflow with Toolhouse

In this notebook you'll learn how to create a sales prospecting workflow using Toolhouse and LlamaIndex. Sales prospecting allows companies to find the perfect potential customer based on the business's value proposition and target market.

The workflow will use a single agent to perform these activities:

1. It will ask the agent to determine a business's value proposition by getting the contents of its landing page.
1. It will search the internet for prospective customers that may benefit from the business's offerings.
1. It will determine the best company to reach out to.
1. It will draft a personalized email to the selected company.

## Initial setup

Let's make sure all the required libraries are present. This example uses Llama 3.2 on Groq, but you can use any the LLMs supported by LlamaIndex.
"""
logger.info("# Sales Prospecting Workflow with Toolhouse")

# %pip install llama-index
# %pip install llama-index-llms-groq
# %pip install toolhouse

"""
Next, we'll pass the API keys.

To get a Toolhouse API key:

1. [Sign up for Toolhouse](https://join.toolhouse.ai) or [sign in](https://app.toolhouse.ai) if you're an existing user.
2. If you're a new user, copy the auto-generated API key you'll receive during onboarding. Existing users can get an API key in the [API Keys page](https://app.toolhouse.ai/settings/api-keys).
3. Paste the API bey below.

To get a Groq API Key, [get access on Groq](https://console.groq.com), then past your API key below.

**Important:** store your API keys safely when in production.
"""
logger.info("Next, we'll pass the API keys.")


os.environ[
    "TOOLHOUSE_API_KEY"
] = "Get your Toolhouse API key at https://join.toolhouse.ai"
os.environ[
    "GROQ_API_KEY"
] = "Get your Groq API key at https://console.groq.com"

"""
## Import libraries

We're going to import LlamaIndexas and Toolhouse. We then initialize Toolhouse and the Groq LLM.
"""
logger.info("## Import libraries")


llm = Groq(model="llama-3.2-11b-vision-preview")

th = Toolhouse(provider=Provider.LLAMAINDEX)
th.set_metadata("id", "llamaindex_agent")
th.set_metadata("timezone", 0)

"""
## Install Toolhouse tools

The agent will require to search the web and get the contents of a page. To allow this, go to your [Toolhouse dashboard](https://app.toolhouse.ai) and install the following tools:

- [Get page contents](https://app.toolhouse.ai/store/scraper)
- [Web search](https://app.toolhouse.ai/store/web_search)

## The Workflow

The workflow will have four steps; we created an output event for each step to make the sequential aspect clearer.

Because Toolhouse integrates directly into LlamaIndex, you can pass the Toolhouse tools directly to the agent.
"""
logger.info("## Install Toolhouse tools")

class WebsiteContentEvent(Event):
    contents: str


class WebSearchEvent(Event):
    results: str


class RankingEvent(Event):
    results: str


class LogEvent(Event):
    msg: str


class SalesRepWorkflow(Workflow):
    agent = ReActAgent(
        tools=th.get_tools(bundle="llamaindex test"),
        llm=llm,
        memory=ChatMemoryBuffer.from_defaults(),
    )

    @step
    async def get_company_info(
        self, ctx: Context, ev: StartEvent
    ) -> WebsiteContentEvent:
        ctx.write_event_to_stream(
            LogEvent(msg=f"Getting the contents of {ev.url}…")
        )
        prompt = f"Get the contents of {ev.url}, then summarize its key value propositions in a few bullet points."
        async def run_async_code_a5b1ffc6():
            async def run_async_code_523b04a8():
                contents = self.agent.chat(prompt)
                return contents
            contents = asyncio.run(run_async_code_523b04a8())
            logger.success(format_json(contents))
            return contents
        contents = asyncio.run(run_async_code_a5b1ffc6())
        logger.success(format_json(contents))
        return WebsiteContentEvent(contents=str(contents.response))

    @step
    async def find_prospects(
        self, ctx: Context, ev: WebsiteContentEvent
    ) -> WebSearchEvent:
        ctx.write_event_to_stream(
            LogEvent(
                msg=f"Performing web searches to identify companies who can benefit from the business's offerings."
            )
        )
        prompt = f"With that you know about the business, perform a web search to find 5 tech companies who may benefit from the business's product. Only answer with the names of the companies you chose."
        async def run_async_code_be5cf2ca():
            async def run_async_code_c45cc5b7():
                results = self.agent.chat(prompt)
                return results
            results = asyncio.run(run_async_code_c45cc5b7())
            logger.success(format_json(results))
            return results
        results = asyncio.run(run_async_code_be5cf2ca())
        logger.success(format_json(results))
        return WebSearchEvent(results=str(results.response))

    @step
    async def select_best_company(
        self, ctx: Context, ev: WebSearchEvent
    ) -> RankingEvent:
        ctx.write_event_to_stream(
            LogEvent(
                msg=f"Selecting the best company who can benefit from the business's offering…"
            )
        )
        prompt = "Select one company that can benefit from the business's product. Only use your knowledge to select the company. Respond with just the name of the company. Do not use tools."
        async def run_async_code_be5cf2ca():
            async def run_async_code_c45cc5b7():
                results = self.agent.chat(prompt)
                return results
            results = asyncio.run(run_async_code_c45cc5b7())
            logger.success(format_json(results))
            return results
        results = asyncio.run(run_async_code_be5cf2ca())
        logger.success(format_json(results))
        ctx.write_event_to_stream(
            LogEvent(
                msg=f"The agent selected this company: {results.response}"
            )
        )
        return RankingEvent(results=str(results.response))

    @step
    async def prepare_email(self, ctx: Context, ev: RankingEvent) -> StopEvent:
        ctx.write_event_to_stream(
            LogEvent(msg=f"Drafting a short email for sales outreach…")
        )
        prompt = f"Draft a short cold sales outreach email for the company you picked. Do not use tools."
        async def run_async_code_f83e7ee4():
            async def run_async_code_0da59345():
                email = self.agent.chat(prompt)
                return email
            email = asyncio.run(run_async_code_0da59345())
            logger.success(format_json(email))
            return email
        email = asyncio.run(run_async_code_f83e7ee4())
        logger.success(format_json(email))
        ctx.write_event_to_stream(
            LogEvent(msg=f"Here is the email: {email.response}")
        )
        return StopEvent(result=str(email.response))

"""
## Run the workflow

Simply instantiate the workflow and pass the URL of a company to get started.
"""
logger.info("## Run the workflow")

workflow = SalesRepWorkflow(timeout=None)
handler = workflow.run(url="https://toolhouse.ai")
async for event in handler.stream_events():
    if isinstance(event, LogEvent):
        logger.debug(event.msg)

logger.info("\n\n[DONE]", bright=True)