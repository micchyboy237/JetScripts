import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
import asyncio
import autogen
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
# Interactive LLM Agent Dealing with Data Stream

AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

In this notebook, we demonstrate how to use customized agents to continuously acquire news from the web and ask for investment suggestions.

## Requirements

AutoGen requires `Python>=3.8`. To run this notebook example, please install:
```bash
pip install autogen
```
"""
logger.info("# Interactive LLM Agent Dealing with Data Stream")



"""
## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("## Set your API Endpoint")



config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

"""
It first looks for environment variable "OAI_CONFIG_LIST" which needs to be a valid json string. If that variable is not found, it then looks for a json file named "OAI_CONFIG_LIST". It filters the configs by models (you can filter by other keys as well). Only the models with matching names are kept in the list based on the filter condition.

The config list looks like the following:
```python
config_list = [
    {
        'model': 'gpt-4',
        'api_key': '<your MLX API key here>',
    },
    {
        'model': 'gpt-3.5-turbo',
        'api_key': '<your Azure MLX API key here>',
        'base_url': '<your Azure MLX API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01',
    },
    {
        'model': 'gpt-3.5-turbo-16k',
        'api_key': '<your Azure MLX API key here>',
        'base_url': '<your Azure MLX API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01',
    },
]
```

You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/autogenhub/autogen/blob/main/website/docs/topics/llm_configuration.ipynb) for full code examples of the different methods.

## Example Task: Investment suggestion with realtime data

We consider a scenario where news data are streamed from a source, and we use an assistant agent to provide investment suggestions based on the data continually.

First, we use the following code to simulate the data stream process.
"""
logger.info("## Example Task: Investment suggestion with realtime data")

def get_market_news(ind, ind_upper):


    data = {
        "feed": [
            {
                "title": "Palantir CEO Says Our Generation's Atomic Bomb Could Be AI Weapon - And Arrive Sooner Than You Think - Palantir Technologies  ( NYSE:PLTR ) ",
                "summary": "Christopher Nolan's blockbuster movie \"Oppenheimer\" has reignited the public discourse surrounding the United States' use of an atomic bomb on Japan at the end of World War II.",
                "overall_sentiment_score": 0.009687,
            },
            {
                "title": '3 "Hedge Fund Hotels" Pulling into Support',
                "summary": "Institutional quality stocks have several benefits including high-liquidity, low beta, and a long runway. Strategist Andrew Rocco breaks down what investors should look for and pitches 3 ideas.",
                "banner_image": "https://staticx-tuner.zacks.com/images/articles/main/92/87.jpg",
                "overall_sentiment_score": 0.219747,
            },
            {
                "title": "PDFgear, Bringing a Completely-Free PDF Text Editing Feature",
                "summary": "LOS ANGELES, July 26, 2023 /PRNewswire/ -- PDFgear, a leading provider of PDF solutions, announced a piece of exciting news for everyone who works extensively with PDF documents.",
                "overall_sentiment_score": 0.360071,
            },
            {
                "title": "Researchers Pitch 'Immunizing' Images Against Deepfake Manipulation",
                "summary": "A team at MIT says injecting tiny disruptive bits of code can cause distorted deepfake images.",
                "overall_sentiment_score": -0.026894,
            },
            {
                "title": "Nvidia wins again - plus two more takeaways from this week's mega-cap earnings",
                "summary": "We made some key conclusions combing through quarterly results for Microsoft and Alphabet and listening to their conference calls with investors.",
                "overall_sentiment_score": 0.235177,
            },
        ]
    }
    feeds = data["feed"][ind:ind_upper]
    feeds_summary = "\n".join(
        [
            f"News summary: {f['title']}. {f['summary']} overall_sentiment_score: {f['overall_sentiment_score']}"
            for f in feeds
        ]
    )
    return feeds_summary


data = asyncio.Future()


async def add_stock_price_data():
    for i in range(0, 5, 1):
        latest_news = get_market_news(i, i + 1)
        if data.done():
            data.result().append(latest_news)
        else:
            data.set_result([latest_news])
        async def run_async_code_baeaee97():
            await asyncio.sleep(5)
            return 
         = asyncio.run(run_async_code_baeaee97())
        logger.success(format_json())


data_task = asyncio.create_task(add_stock_price_data())

"""
Then, we construct agents. An assistant agent is created to answer the question using LLM. A UserProxyAgent is created to ask questions, and add the new data in the conversation when they are available.
"""
logger.info("Then, we construct agents. An assistant agent is created to answer the question using LLM. A UserProxyAgent is created to ask questions, and add the new data in the conversation when they are available.")

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "timeout": 600,
        "cache_seed": 41,
        "config_list": config_list,
        "temperature": 0,
    },
    system_message="You are a financial expert.",
)
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    code_execution_config=False,
    default_auto_reply=None,
)


async def add_data_reply(recipient, messages, sender, config):
    async def run_async_code_09c0cbe8():
        await asyncio.sleep(0.1)
        return 
     = asyncio.run(run_async_code_09c0cbe8())
    logger.success(format_json())
    data = config["news_stream"]
    if data.done():
        result = data.result()
        if result:
            news_str = "\n".join(result)
            result.clear()
            return (
                True,
                f"Just got some latest market news. Merge your new suggestion with previous ones.\n{news_str}",
            )
        return False, None


user_proxy.register_reply(autogen.AssistantAgent, add_data_reply, position=2, config={"news_stream": data})

"""
We invoke the `a_initiate_chat()` method of the user proxy agent to start the conversation.
"""
logger.info("We invoke the `a_initiate_chat()` method of the user proxy agent to start the conversation.")

async def run_async_code_4606133e():
    await user_proxy.a_initiate_chat(  # noqa: F704
    return 
 = asyncio.run(run_async_code_4606133e())
logger.success(format_json())
    assistant,
    message="""Give me investment suggestion in 3 bullet points.""",
)
while not data_task.done() and not data_task.cancelled():
    async def run_async_code_ebd852b2():
        async def run_async_code_c3926cbc():
            reply = await user_proxy.a_generate_reply(sender=assistant)  # noqa: F704
            return reply
        reply = asyncio.run(run_async_code_c3926cbc())
        logger.success(format_json(reply))
        return reply
    reply = asyncio.run(run_async_code_ebd852b2())
    logger.success(format_json(reply))
    if reply is not None:
        async def run_async_code_e77d88be():
            await user_proxy.a_send(reply, assistant)  # noqa: F704
            return 
         = asyncio.run(run_async_code_e77d88be())
        logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)