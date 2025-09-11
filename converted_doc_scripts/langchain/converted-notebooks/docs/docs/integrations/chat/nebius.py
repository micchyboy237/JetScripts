from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nebius import ChatNebius
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
    ---
    sidebar_label: Nebius
    ---
    
    # Nebius Chat Models
    
    This page will help you get started with Nebius AI Studio [chat models](../../concepts/chat_models.mdx). For detailed documentation of all ChatNebius features and configurations head to the [API reference](https://python.langchain.com/api_reference/nebius/chat_models/langchain_nebius.chat_models.ChatNebius.html).
    
    [Nebius AI Studio](https://studio.nebius.ai/) provides API access to a wide range of state-of-the-art large language models and embedding models for various use cases.
    
    ## Overview
    
    ### Integration details
    
    | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
    | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
    | [ChatNebius](https://python.langchain.com/api_reference/nebius/chat_models/langchain_nebius.chat_models.ChatNebius.html) | [langchain-nebius](https://python.langchain.com/api_reference/nebius/index.html) | ❌ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-nebius?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-nebius?style=flat-square&label=%20) |
    
    ### Model features
    | [Tool calling](../../how_to/tool_calling.ipynb) | [Structured output](../../how_to/structured_output.ipynb) | JSON mode | [Image input](../../how_to/multimodal_inputs.ipynb) | Audio input | Video input | [Token-level streaming](../../how_to/chat_streaming.ipynb) | Native async | [Token usage](../../how_to/chat_token_usage_tracking.ipynb) | [Logprobs](../../how_to/logprobs.ipynb) |
    | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
    | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
    
    ## Setup
    
    To access Nebius models you'll need to create a Nebius account, get an API key, and install the `langchain-nebius` integration package.
    
    ### Installation
    
    The Nebius integration can be installed via pip:
    """
    logger.info("# Nebius Chat Models")
    
    # %pip install --upgrade langchain-nebius
    
    """
    ### Credentials
    
    Nebius requires an API key that can be passed as an initialization parameter `api_key` or set as the environment variable `NEBIUS_API_KEY`. You can obtain an API key by creating an account on [Nebius AI Studio](https://studio.nebius.ai/).
    """
    logger.info("### Credentials")
    
    # import getpass
    
    if "NEBIUS_API_KEY" not in os.environ:
    #     os.environ["NEBIUS_API_KEY"] = getpass.getpass("Enter your Nebius API key: ")
    
    """
    ## Instantiation
    
    Now we can instantiate our model object to generate chat completions:
    """
    logger.info("## Instantiation")
    
    
    chat = ChatNebius(
        model="Qwen/Qwen3-14B",  # Choose from available models
        temperature=0.6,
        top_p=0.95,
    )
    
    """
    ## Invocation
    
    You can use the `invoke` method to get a completion from the model:
    """
    logger.info("## Invocation")
    
    response = chat.invoke("Explain quantum computing in simple terms")
    logger.debug(response.content)
    
    """
    ### Streaming
    
    You can also stream the response using the `stream` method:
    """
    logger.info("### Streaming")
    
    for chunk in chat.stream("Write a short poem about artificial intelligence"):
        logger.debug(chunk.content, end="", flush=True)
    
    """
    ### Chat Messages
    
    You can use different message types to structure your conversations with the model:
    """
    logger.info("### Chat Messages")
    
    
    messages = [
        SystemMessage(content="You are a helpful AI assistant with expertise in science."),
        HumanMessage(content="What are black holes?"),
        AIMessage(
            content="Black holes are regions of spacetime where gravity is so strong that nothing, including light, can escape from them."
        ),
        HumanMessage(content="How are they formed?"),
    ]
    
    response = chat.invoke(messages)
    logger.debug(response.content)
    
    """
    ### Parameters
    
    You can customize the chat model behavior using various parameters:
    """
    logger.info("### Parameters")
    
    custom_chat = ChatNebius(
        model="meta-llama/Llama-3.3-70B-Instruct-fast",
        max_tokens=100,  # Limit response length
        top_p=0.01,  # Lower nucleus sampling parameter for more deterministic responses
        request_timeout=30,  # Timeout in seconds
        stop=["###", "\n\n"],  # Custom stop sequences
    )
    
    response = custom_chat.invoke("Explain what DNA is in exactly 3 sentences.")
    logger.debug(response.content)
    
    """
    You can also pass parameters at invocation time:
    """
    logger.info("You can also pass parameters at invocation time:")
    
    standard_chat = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast")
    
    response = standard_chat.invoke(
        "Tell me a joke about programming",
        temperature=0.9,  # More creative for jokes
        max_tokens=50,  # Keep it short
    )
    
    logger.debug(response.content)
    
    """
    ### Async Support
    
    ChatNebius supports async operations:
    """
    logger.info("### Async Support")
    
    
    
    async def generate_async():
        response = await chat.ainvoke("What is the capital of France?")
        logger.success(format_json(response))
        logger.debug("Async response:", response.content)
    
        logger.debug("\nAsync streaming:")
        for chunk in chat.stream("What is the capital of Germany?"):
            logger.debug(chunk.content, end="", flush=True)
    
    
    await generate_async()
    
    """
    ### Available Models
    
    The full list of supported models can be found in the [Nebius AI Studio Documentation](https://studio.nebius.com/).
    
    ## Chaining
    
    You can use `ChatNebius` in LangChain chains and agents:
    """
    logger.info("### Available Models")
    
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers in the style of {character}.",
            ),
            ("human", "{query}"),
        ]
    )
    
    chain = prompt | chat | StrOutputParser()
    
    response = chain.invoke(
        {"character": "Shakespeare", "query": "Explain how the internet works"}
    )
    
    logger.debug(response)
    
    """
    ## API reference
    
    For more details about the Nebius AI Studio API, visit the [Nebius AI Studio Documentation](https://studio.nebius.com/api-reference).
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