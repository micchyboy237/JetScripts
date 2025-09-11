from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_goodfire import ChatGoodfire
import goodfire
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
    sidebar_label: Goodfire
    ---
    
    # ChatGoodfire
    
    This will help you get started with Goodfire [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatGoodfire features and configurations head to the [PyPI project page](https://pypi.org/project/langchain-goodfire/), or go directly to the [Goodfire SDK docs](https://docs.goodfire.ai/sdk-reference/example). All of the Goodfire-specific functionality (e.g. SAE features, variants, etc.) is available via the main `goodfire` package. This integration is a wrapper around the Goodfire SDK.
    
    ## Overview
    ### Integration details
    
    | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
    | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
    | [ChatGoodfire](https://python.langchain.com/api_reference/goodfire/chat_models/langchain_goodfire.chat_models.ChatGoodfire.html) | [langchain-goodfire](https://python.langchain.com/api_reference/goodfire/) | ❌ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-goodfire?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-goodfire?style=flat-square&label=%20) |
    
    ### Model features
    | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
    | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
    | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
    
    ## Setup
    
    To access Goodfire models you'll need to create a/an Goodfire account, get an API key, and install the `langchain-goodfire` integration package.
    
    ### Credentials
    
    Head to [Goodfire Settings](https://platform.goodfire.ai/organization/settings/api-keys) to sign up to Goodfire and generate an API key. Once you've done this set the GOODFIRE_API_KEY environment variable.
    """
    logger.info("# ChatGoodfire")
    
    # import getpass
    
    if not os.getenv("GOODFIRE_API_KEY"):
    #     os.environ["GOODFIRE_API_KEY"] = getpass.getpass("Enter your Goodfire API key: ")
    
    """
    T
    o
     
    e
    n
    a
    b
    l
    e
     
    a
    u
    t
    o
    m
    a
    t
    e
    d
     
    t
    r
    a
    c
    i
    n
    g
     
    o
    f
     
    y
    o
    u
    r
     
    m
    o
    d
    e
    l
     
    c
    a
    l
    l
    s
    ,
     
    s
    e
    t
     
    y
    o
    u
    r
     
    [
    L
    a
    n
    g
    S
    m
    i
    t
    h
    ]
    (
    h
    t
    t
    p
    s
    :
    /
    /
    d
    o
    c
    s
    .
    s
    m
    i
    t
    h
    .
    l
    a
    n
    g
    c
    h
    a
    i
    n
    .
    c
    o
    m
    /
    )
     
    A
    P
    I
     
    k
    e
    y
    :
    """
    logger.info("T")
    
    
    
    """
    ### Installation
    
    The LangChain Goodfire integration lives in the `langchain-goodfire` package:
    """
    logger.info("### Installation")
    
    # %pip install -qU langchain-goodfire
    
    """
    ## Instantiation
    
    Now we can instantiate our model object and generate chat completions:
    """
    logger.info("## Instantiation")
    
    
    base_variant = goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
    
    llm = ChatGoodfire(
        model=base_variant,
        temperature=0,
        max_completion_tokens=1000,
        seed=42,
    )
    
    """
    ## Invocation
    """
    logger.info("## Invocation")
    
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = await llm.ainvoke(messages)
    logger.success(format_json(ai_msg))
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
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )
    
    chain = prompt | llm
    await chain.ainvoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    
    """
    ## Goodfire-specific functionality
    
    To use Goodfire-specific functionality such as SAE features and variants, you can use the `goodfire` package directly.
    """
    logger.info("## Goodfire-specific functionality")
    
    client = goodfire.Client(api_key=os.environ["GOODFIRE_API_KEY"])
    
    pirate_features = client.features.search(
        "assistant should roleplay as a pirate", base_variant
    )
    pirate_features
    
    pirate_variant = goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
    
    pirate_variant.set(pirate_features[0], 0.4)
    pirate_variant.set(pirate_features[1], 0.3)
    
    await llm.ainvoke("Tell me a joke", model=pirate_variant)
    
    """
    ## API reference
    
    For detailed documentation of all ChatGoodfire features and configurations head to the [API reference](https://python.langchain.com/api_reference/goodfire/chat_models/langchain_goodfire.chat_models.ChatGoodfire.html)
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