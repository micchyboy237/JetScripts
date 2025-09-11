from datetime import datetime
from jet.logger import logger
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_sambanova import ChatSambaNovaCloud
from pydantic import BaseModel, Field
import base64
import httpx
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
    ---
    sidebar_label: SambaNovaCloud
    ---
    
    # ChatSambaNovaCloud
    
    This will help you get started with SambaNovaCloud [chat models](/docs/concepts/chat_models/). For detailed documentation of all ChatSambaNovaCloud features and configurations head to the [API reference](https://docs.sambanova.ai/cloud/docs/get-started/overview).
    
    **[SambaNova](https://sambanova.ai/)'s** [SambaNova Cloud](https://cloud.sambanova.ai/) is a platform for performing inference with open-source models
    
    ## Overview
    ### Integration details
    
    | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
    | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
    | [ChatSambaNovaCloud](https://docs.sambanova.ai/cloud/docs/get-started/overview) | [langchain-sambanova](https://python.langchain.com/docs/integrations/providers/sambanova/) | ❌ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_sambanova?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_sambanova?style=flat-square&label=%20) |
    
    ### Model features
    
    | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](//docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
    | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
    | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | 
    
    ## Setup
    
    To access ChatSambaNovaCloud models you will need to create a [SambaNovaCloud](https://cloud.sambanova.ai/) account, get an API key, install the `langchain_sambanova` integration package.
    
    ```bash
    pip install langchain-sambanova
    ```
    
    ### Credentials
    
    Get an API Key from [cloud.sambanova.ai](https://cloud.sambanova.ai/apis) and add it to your environment variables:
    
    ``` bash
    export SAMBANOVA_API_KEY="your-api-key-here"
    ```
    """
    logger.info("# ChatSambaNovaCloud")
    
    # import getpass
    
    if not os.getenv("SAMBANOVA_API_KEY"):
    #     os.environ["SAMBANOVA_API_KEY"] = getpass.getpass(
            "Enter your SambaNova Cloud API key: "
        )
    
    """
    If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
    """
    logger.info("If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")
    
    
    
    """
    ### Installation
    
    The LangChain __SambaNovaCloud__ integration lives in the `langchain_sambanova` package:
    """
    logger.info("### Installation")
    
    # %pip install -qU langchain-sambanova
    
    """
    ## Instantiation
    
    Now we can instantiate our model object and generate chat completions:
    """
    logger.info("## Instantiation")
    
    
    llm = ChatSambaNovaCloud(
        model="Meta-Llama-3.3-70B-Instruct",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.01,
    )
    
    """
    ## Invocation
    """
    logger.info("## Invocation")
    
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. "
            "Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)
    ai_msg
    
    logger.debug(ai_msg.content)
    
    """
    ## Chaining
    
    We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:
    """
    logger.info("## Chaining")
    
    
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} "
                "to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )
    
    chain = prompt | llm
    chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    
    """
    ## Streaming
    """
    logger.info("## Streaming")
    
    system = "You are a helpful assistant with pirate accent."
    human = "I want to learn more about this animal: {animal}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    
    chain = prompt | llm
    
    for chunk in chain.stream({"animal": "owl"}):
        logger.debug(chunk.content, end="", flush=True)
    
    """
    ## Async
    """
    logger.info("## Async")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "what is the capital of {country}?",
            )
        ]
    )
    
    chain = prompt | llm
    await chain.ainvoke({"country": "France"})
    
    """
    ## Async Streaming
    """
    logger.info("## Async Streaming")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "in less than {num_words} words explain me {topic} ",
            )
        ]
    )
    chain = prompt | llm
    
    for chunk in chain.stream({"num_words": 30, "topic": "quantum computers"}):
        logger.debug(chunk.content, end="", flush=True)
    
    """
    ## Tool calling
    """
    logger.info("## Tool calling")
    
    
    
    
    @tool
    def get_time(kind: str = "both") -> str:
        """Returns current date, current time or both.
        Args:
            kind(str): date, time or both
        Returns:
            str: current date, current time or both
        """
        if kind == "date":
            date = datetime.now().strftime("%m/%d/%Y")
            return f"Current date: {date}"
        elif kind == "time":
            time = datetime.now().strftime("%H:%M:%S")
            return f"Current time: {time}"
        else:
            date = datetime.now().strftime("%m/%d/%Y")
            time = datetime.now().strftime("%H:%M:%S")
            return f"Current date: {date}, Current time: {time}"
    
    
    tools = [get_time]
    
    
    def invoke_tools(tool_calls, messages):
        available_functions = {tool.name: tool for tool in tools}
        for tool_call in tool_calls:
            selected_tool = available_functions[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            logger.debug(f"Tool output: {tool_output}")
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
        return messages
    
    llm_with_tools = llm.bind_tools(tools=tools)
    messages = [
        HumanMessage(
            content="I need to schedule a meeting for two weeks from today. "
            "Can you tell me the exact date of the meeting?"
        )
    ]
    
    response = llm_with_tools.invoke(messages)
    while len(response.tool_calls) > 0:
        logger.debug(f"Intermediate model response: {response.tool_calls}")
        messages.append(response)
        messages = invoke_tools(response.tool_calls, messages)
        response = llm_with_tools.invoke(messages)
    
    logger.debug(f"final response: {response.content}")
    
    """
    ## Structured Outputs
    """
    logger.info("## Structured Outputs")
    
    
    
    class Joke(BaseModel):
        """Joke to tell user."""
    
        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")
    
    
    structured_llm = llm.with_structured_output(Joke)
    
    structured_llm.invoke("Tell me a joke about cats")
    
    """
    ## Input Image
    """
    logger.info("## Input Image")
    
    multimodal_llm = ChatSambaNovaCloud(
        model="Llama-3.2-11B-Vision-Instruct",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.01,
    )
    
    
    
    image_url = (
        "https://images.pexels.com/photos/147411/italy-mountains-dawn-daybreak-147411.jpeg"
    )
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe the weather in this image in 1 sentence"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ],
    )
    response = multimodal_llm.invoke([message])
    logger.debug(response.content)
    
    """
    ## API reference
    
    For detailed documentation of all SambaNovaCloud features and configurations head to the API reference: https://docs.sambanova.ai/cloud/docs/get-started/overview
    """
    logger.info("## API reference")
    
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