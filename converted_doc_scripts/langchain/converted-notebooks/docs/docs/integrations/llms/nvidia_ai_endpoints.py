from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import NVIDIA
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
    # NVIDIA
    
    This will help you get started with NVIDIA [models](/docs/concepts/text_llms). For detailed documentation of all `NVIDIA` features and configurations head to the [API reference](https://python.langchain.com/api_reference/nvidia_ai_endpoints/llms/langchain_nvidia_ai_endpoints.chat_models.NVIDIA.html).
    
    ## Overview
    The `langchain-nvidia-ai-endpoints` package contains LangChain integrations building applications with models on 
    NVIDIA NIM inference microservice. These models are optimized by NVIDIA to deliver the best performance on NVIDIA 
    accelerated infrastructure and deployed as a NIM, an easy-to-use, prebuilt containers that deploy anywhere using a single 
    command on NVIDIA accelerated infrastructure.
    
    NVIDIA hosted deployments of NIMs are available to test on the [NVIDIA API catalog](https://build.nvidia.com/). After testing, 
    NIMs can be exported from NVIDIA’s API catalog using the NVIDIA AI Enterprise license and run on-premises or in the cloud, 
    giving enterprises ownership and full control of their IP and AI application.
    
    NIMs are packaged as container images on a per model basis and are distributed as NGC container images through the NVIDIA NGC Catalog. 
    At their core, NIMs provide easy, consistent, and familiar APIs for running inference on an AI model.
    
    This example goes over how to use LangChain to interact with NVIDIA supported via the `NVIDIA` class.
    
    For more information on accessing the llm models through this api, check out the [NVIDIA](https://python.langchain.com/docs/integrations/llms/nvidia_ai_endpoints/) documentation.
    
    ### Integration details
    
    | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
    | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
    | [NVIDIA](https://python.langchain.com/api_reference/nvidia_ai_endpoints/llms/langchain_nvidia_ai_endpoints.chat_models.ChatNVIDIA.html) | [langchain-nvidia-ai-endpoints](https://python.langchain.com/api_reference/nvidia_ai_endpoints/index.html) | ✅ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_nvidia_ai_endpoints?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_nvidia_ai_endpoints?style=flat-square&label=%20) |
    
    ### Model features
    | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
    | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
    | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 
    
    ## Setup
    
    **To get started:**
    
    1. Create a free account with [NVIDIA](https://build.nvidia.com/), which hosts NVIDIA AI Foundation models.
    
    2. Click on your model of choice.
    
    3. Under `Input` select the `Python` tab, and click `Get API Key`. Then click `Generate Key`.
    
    4. Copy and save the generated key as `NVIDIA_API_KEY`. From there, you should have access to the endpoints.
    
    ### Credentials
    """
    logger.info("# NVIDIA")
    
    # import getpass
    
    if not os.getenv("NVIDIA_API_KEY"):
    #     os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter your NVIDIA API key: ")
    
    """
    ### Installation
    
    The LangChain NVIDIA AI Endpoints integration lives in the `langchain-nvidia-ai-endpoints` package:
    """
    logger.info("### Installation")
    
    # %pip install --upgrade --quiet langchain-nvidia-ai-endpoints
    
    """
    ## Instantiation
    
    See [LLM](/docs/how_to#llms) for full functionality.
    """
    logger.info("## Instantiation")
    
    
    llm = NVIDIA().bind(max_tokens=256)
    llm
    
    """
    ## Invocation
    """
    logger.info("## Invocation")
    
    prompt = "# Function that does quicksort written in Rust without comments:"
    
    logger.debug(llm.invoke(prompt))
    
    """
    ## Stream, Batch, and Async
    
    These models natively support streaming, and as is the case with all LangChain LLMs they expose a batch method to handle concurrent requests, as well as async methods for invoke, stream, and batch. Below are a few examples.
    """
    logger.info("## Stream, Batch, and Async")
    
    for chunk in llm.stream(prompt):
        logger.debug(chunk, end="", flush=True)
    
    llm.batch([prompt])
    
    await llm.ainvoke(prompt)
    
    for chunk in llm.stream(prompt):
        logger.debug(chunk, end="", flush=True)
    
    await llm.abatch([prompt])
    
    for chunk in llm.stream_log(prompt):
        logger.debug(chunk)
    
    response = llm.invoke(
        "X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.1) #Train a logistic regression model, predict the labels on the test set and compute the accuracy score"
    )
    logger.debug(response)
    
    """
    ## Supported models
    
    Querying `available_models` will still give you all of the other models offered by your API credentials.
    """
    logger.info("## Supported models")
    
    NVIDIA.get_available_models()
    
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
    chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    
    """
    ## API reference
    
    For detailed documentation of all `NVIDIA` features and configurations head to the API reference: https://python.langchain.com/api_reference/nvidia_ai_endpoints/llms/langchain_nvidia_ai_endpoints.llms.NVIDIA.html
    
    
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