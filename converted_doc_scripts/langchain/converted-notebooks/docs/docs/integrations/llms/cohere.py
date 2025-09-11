from jet.logger import logger
from langchain_cohere import Cohere
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
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
    # Cohere
    
    :::caution
    You are currently on a page documenting the use of Cohere models as [text completion models](/docs/concepts/text_llms). Many popular Cohere models are [chat completion models](/docs/concepts/chat_models).
    
    You may be looking for [this page instead](/docs/integrations/chat/cohere/).
    :::
    
    >[Cohere](https://cohere.ai/about) is a Canadian startup that provides natural language processing models that help companies improve human-machine interactions.
    
    Head to the [API reference](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.cohere.Cohere.html) for detailed documentation of all attributes and methods.
    
    ## Overview
    ### Integration details
    
    | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/llms/cohere/) | Package downloads | Package latest |
    | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
    | [Cohere](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.cohere.Cohere.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |
    
    ## Setup
    
    The integration lives in the `langchain-community` package. We also need to install the `cohere` package itself. We can install these with:
    
    ### Credentials
    
    We'll need to get a [Cohere API key](https://cohere.com/) and set the `COHERE_API_KEY` environment variable:
    """
    logger.info("# Cohere")
    
    # import getpass
    
    if "COHERE_API_KEY" not in os.environ:
    #     os.environ["COHERE_API_KEY"] = getpass.getpass()
    
    """
    ### Installation
    """
    logger.info("### Installation")
    
    pip install -U langchain-community langchain-cohere
    
    """
    It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability
    """
    logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability")
    
    
    
    """
    ## Invocation
    
    Cohere supports all [LLM](/docs/how_to#llms) functionality:
    """
    logger.info("## Invocation")
    
    
    model = Cohere(max_tokens=256, temperature=0.75)
    
    message = "Knock knock"
    model.invoke(message)
    
    await model.ainvoke(message)
    
    for chunk in model.stream(message):
        logger.debug(chunk, end="", flush=True)
    
    model.batch([message])
    
    """
    ## Chaining
    
    You can also easily combine with a prompt template for easy structuring of user input. We can do this using [LCEL](/docs/concepts/lcel)
    """
    logger.info("## Chaining")
    
    
    prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
    chain = prompt | model
    
    chain.invoke({"topic": "bears"})
    
    """
    ## API reference
    
    For detailed documentation of all `Cohere` llm features and configurations head to the API reference: https://python.langchain.com/api_reference/community/llms/langchain_community.llms.cohere.Cohere.html
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