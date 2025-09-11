from jet.transformers.formatters import format_json
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_chroma import Chroma
from langchain_core.output_parsers.ollama_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List, Optional
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
    sidebar_position: 4
    ---
    
    # How to handle multiple queries when doing query analysis
    
    Sometimes, a query analysis technique may allow for multiple queries to be generated. In these cases, we need to remember to run all queries and then to combine the results. We will show a simple example (using mock data) of how to do that.
    
    ## Setup
    #### Install dependencies
    """
    logger.info("# How to handle multiple queries when doing query analysis")

    # %pip install -qU langchain langchain-community langchain-ollama langchain-chroma

    """
    #### Set environment variables
    
    We'll use Ollama in this example:
    """
    logger.info("#### Set environment variables")

    # import getpass

    # if "OPENAI_API_KEY" not in os.environ:
    #     os.environ["OPENAI_API_KEY"] = getpass.getpass()

    """
    ### Create Index
    
    We will create a vectorstore over fake information.
    """
    logger.info("### Create Index")

    texts = ["Harrison worked at Kensho", "Ankush worked at Facebook"]
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_texts(
        texts,
        embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    """
    ## Query analysis
    
    We will use function calling to structure the output. We will let it return multiple queries.
    """
    logger.info("## Query analysis")

    class Search(BaseModel):
        """Search over a database of job records."""

        queries: List[str] = Field(
            ...,
            description="Distinct queries to search for",
        )

    output_parser = PydanticToolsParser(tools=[Search])

    system = """You have the ability to issue search queries to get information to help answer user information.
    
    If you need to look up two distinct pieces of information, you are allowed to do that!"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm = ChatOllama(model="llama3.2")
    structured_llm = llm.with_structured_output(Search)
    query_analyzer = {
        "question": RunnablePassthrough()} | prompt | structured_llm

    """
    We can see that this allows for creating multiple queries
    """
    logger.info("We can see that this allows for creating multiple queries")

    query_analyzer.invoke("where did Harrison Work")

    query_analyzer.invoke("where did Harrison and ankush Work")

    """
    ## Retrieval with query analysis
    
    So how would we include this in a chain? One thing that will make this a lot easier is if we call our retriever asynchronously - this will let us loop over the queries and not get blocked on the response time.
    """
    logger.info("## Retrieval with query analysis")

    @chain
    async def custom_chain(question):
        response = await query_analyzer.ainvoke(question)
        logger.success(format_json(response))
        docs = []
        for query in response.queries:
            new_docs = await retriever.ainvoke(query)
            logger.success(format_json(new_docs))
            docs.extend(new_docs)
        return docs

    await custom_chain.ainvoke("where did Harrison Work")

    await custom_chain.ainvoke("where did Harrison and ankush Work")

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
