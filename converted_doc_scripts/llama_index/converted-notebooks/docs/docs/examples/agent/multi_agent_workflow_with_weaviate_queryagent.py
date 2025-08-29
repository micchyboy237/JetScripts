async def main():
    from jet.transformers.formatters import format_json
    from enum import Enum
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapterResponses
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.llms import ChatMessage
    from llama_index.core.tools import FunctionTool
    from llama_index.core.workflow import (
        StartEvent,
        StopEvent,
        Workflow,
        step,
        Event,
        Context,
    )
    from llama_index.readers.web import SimpleWebPageReader
    from llama_index.utils.workflow import draw_all_possible_flows
    from pydantic import BaseModel, Field
    from typing import List, Union
    from weaviate.agents.query import QueryAgent
    from weaviate.auth import Auth
    from weaviate.classes.config import Configure, Property, DataType
    import json
    import os
    import shutil
    import weaviate

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    # Multi-Agent Workflow with Weaviate QueryAgent
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/multi_agent_workflow_with_weaviate_queryagent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    In this example, we will be building a LlamaIndex Agent Workflow that ends up being a multi-agent system that aims to be a Docs Assistant capable of:
    - Writing new content to a "LlamaIndexDocs" collection in Weaviate
    - Writing new content to a "WeaviateDocs" collection in Weaviate
    - Using the Weaviate [`QueryAgent`](https://weaviate.io/developers/agents/query) to answer questions based on the contents of these collections.
    
    The `QueryAgent` is a full agent prodcut by Weaviate, that is capable of doing regular search, as well as aggregations over the collections you give it access to. Our 'orchestrator' agent will decide when to invoke the Weaviate QueryAgent, leaving the job of creating Weaviate specific search queries to it.
    
    **Things you will need:**
    
    - An OllamaFunctionCallingAdapter API key (or switch to another provider and adjust the code below)
    - A Weaviate sandbox (this is free)
    - Your Weaviate sandbox URL and API key
    
    ![Workflow Overview](../../../_static/agents/workflow-weaviate-multiagent.png)
    
    ## Install & Import Dependencies
    """
    logger.info("# Multi-Agent Workflow with Weaviate QueryAgent")

    # !pip install llama-index-core llama-index-utils-workflow weaviate-client[agents] llama-index-llms-ollama llama-index-readers-web

    # from getpass import getpass

    """
    ## Set up Weaviate
    
    To use the Weaviate Query Agent, first, create a [Weaviate Cloud](https://weaviate.io/deployment/serverless) account👇
    1. [Create Serverless Weaviate Cloud account](https://weaviate.io/deployment/serverless) and set up a free [Sandbox](https://weaviate.io/developers/wcs/manage-clusters/create#sandbox-clusters)
    2. Go to 'Embedding' and enable it, by default, this will make it so that we use `Snowflake/snowflake-arctic-embed-l-v2.0` as the embedding model
    3. Take note of the `WEAVIATE_URL` and `WEAVIATE_API_KEY` to connect to your cluster below
    
    > Info: We recommend using [Weaviate Embeddings](https://weaviate.io/developers/weaviate/model-providers/weaviate) so you do not have to provide any extra keys for external embedding providers.
    """
    logger.info("## Set up Weaviate")

    if "WEAVIATE_API_KEY" not in os.environ:
        #     os.environ["WEAVIATE_API_KEY"] = getpass("Add Weaviate API Key")
    if "WEAVIATE_URL" not in os.environ:
        #     os.environ["WEAVIATE_URL"] = getpass("Add Weaviate URL")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ.get("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.environ.get("WEAVIATE_API_KEY")),
    )

    """
    ### Create WeaviateDocs and LlamaIndexDocs Collections
    
    The helper function below will create a "WeaviateDocs" and "LlamaIndexDocs" collection in Weaviate (if they don't exist already). It will also set up a `QueryAgent` that has access to both of these collections.
    
    The Weaviate [`QueryAgent`](https://weaviate.io/blog/query-agent) is designed to be able to query Weviate Collections for both regular search and aggregations, and also handles the burden of creating the Weaviate specific queries internally.
    
    The Agent will use the collection descriptions, as well as the property descriptions while formilating the queries.
    """
    logger.info("### Create WeaviateDocs and LlamaIndexDocs Collections")

    def fresh_setup_weaviate(client):
        if client.collections.exists("WeaviateDocs"):
            client.collections.delete("WeaviateDocs")
        client.collections.create(
            "WeaviateDocs",
            description="A dataset with the contents of Weaviate technical Docs and website",
            vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
            properties=[
                Property(
                    name="url",
                    data_type=DataType.TEXT,
                    description="the source URL of the webpage",
                ),
                Property(
                    name="text",
                    data_type=DataType.TEXT,
                    description="the content of the webpage",
                ),
            ],
        )

        if client.collections.exists("LlamaIndexDocs"):
            client.collections.delete("LlamaIndexDocs")
        client.collections.create(
            "LlamaIndexDocs",
            description="A dataset with the contents of LlamaIndex technical Docs and website",
            vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
            properties=[
                Property(
                    name="url",
                    data_type=DataType.TEXT,
                    description="the source URL of the webpage",
                ),
                Property(
                    name="text",
                    data_type=DataType.TEXT,
                    description="the content of the webpage",
                ),
            ],
        )

        agent = QueryAgent(
            client=client, collections=["LlamaIndexDocs", "WeaviateDocs"]
        )
        return agent

    """
    ### Write Contents of Webpage to the Collections
    
    The helper function below uses the `SimpleWebPageReader` to write the contents of a webpage to the relevant Weaviate collection
    """
    logger.info("### Write Contents of Webpage to the Collections")

    def write_webpages_to_weaviate(client, urls: list[str], collection_name: str):
        documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
        collection = client.collections.get(collection_name)
        with collection.batch.dynamic() as batch:
            for doc in documents:
                batch.add_object(properties={"url": doc.id_, "text": doc.text})

    """
    ## Create a Function Calling Agent
    
    Now that we have the relevant functions to write to a collection and also the `QueryAgent` at hand, we can start by using the `FunctionAgent`, which is a simple tool calling agent.
    """
    logger.info("## Create a Function Calling Agent")

    # if "OPENAI_API_KEY" not in os.environ:
    #     os.environ["OPENAI_API_KEY"] = getpass("openai-key")

    weaviate_agent = fresh_setup_weaviate(client)

    llm = OllamaFunctionCallingAdapter(model="llama3.2")

    def write_to_weaviate_collection(urls=list[str]):
        """Useful for writing new content to the WeaviateDocs collection"""
        write_webpages_to_weaviate(client, urls, "WeaviateDocs")

    def write_to_li_collection(urls=list[str]):
        """Useful for writing new content to the LlamaIndexDocs collection"""
        write_webpages_to_weaviate(client, urls, "LlamaIndexDocs")

    def query_agent(query: str) -> str:
        """Useful for asking questions about Weaviate and LlamaIndex"""
        response = weaviate_agent.run(query)
        return response.final_answer

    agent = FunctionAgent(
        tools=[write_to_weaviate_collection,
               write_to_li_collection, query_agent],
        llm=llm,
        system_prompt="""You are a helpful assistant that can write the
          contents of urls to WeaviateDocs and LlamaIndexDocs collections,
          as well as forwarding questions to a QueryAgent""",
    )

    response = await agent.run(
        user_msg="Can you save https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/"
    )
    logger.success(format_json(response))
    logger.debug(str(response))

    response = await agent.run(
        user_msg="""What are llama index workflows? And can you save
            these to weaviate docs: https://weaviate.io/blog/what-are-agentic-workflows
            and https://weaviate.io/blog/ai-agents"""
    )
    logger.success(format_json(response))
    logger.debug(str(response))

    response = await agent.run(
        user_msg="How many docs do I have in the weaviate and llamaindex collections in total?"
    )
    logger.success(format_json(response))
    logger.debug(str(response))

    weaviate_agent = fresh_setup_weaviate(client)

    """
    ## Create a Workflow with Branches
    
    ### Simple Example: Create Events
    
    A LlamaIndex Workflow has 2 fundamentals:
    - An Event
    - A Step
    
    An step may return an event, and an event may trigger a step!
    
    For our use-case, we can imagine thet there are 4 events:
    """
    logger.info("## Create a Workflow with Branches")

    class EvaluateQuery(Event):
        query: str

    class WriteLlamaIndexDocsEvent(Event):
        urls: list[str]

    class WriteWeaviateDocsEvent(Event):
        urls: list[str]

    class QueryAgentEvent(Event):
        query: str

    """
    ### Simple Example: A Branching Workflow (that does nothing yet)
    """
    logger.info(
        "### Simple Example: A Branching Workflow (that does nothing yet)")

    class DocsAssistantWorkflow(Workflow):
        @step
        async def start(self, ctx: Context, ev: StartEvent) -> EvaluateQuery:
            return EvaluateQuery(query=ev.query)

        @step
        async def evaluate_query(
            self, ctx: Context, ev: EvaluateQuery
        ) -> QueryAgentEvent | WriteLlamaIndexDocsEvent | WriteWeaviateDocsEvent | StopEvent:
            if ev.query == "llama":
                return WriteLlamaIndexDocsEvent(urls=[ev.query])
            if ev.query == "weaviate":
                return WriteWeaviateDocsEvent(urls=[ev.query])
            if ev.query == "question":
                return QueryAgentEvent(query=ev.query)
            return StopEvent()

        @step
        async def write_li_docs(
            self, ctx: Context, ev: WriteLlamaIndexDocsEvent
        ) -> StopEvent:
            logger.debug(f"Got a request to write something to LlamaIndexDocs")
            return StopEvent()

        @step
        async def write_weaviate_docs(
            self, ctx: Context, ev: WriteWeaviateDocsEvent
        ) -> StopEvent:
            logger.debug(f"Got a request to write something to WeaviateDocs")
            return StopEvent()

        @step
        async def query_agent(
            self, ctx: Context, ev: QueryAgentEvent
        ) -> StopEvent:
            logger.debug(f"Got a request to forward a query to the QueryAgent")
            return StopEvent()

    workflow_that_does_nothing = DocsAssistantWorkflow()

    logger.debug(
        await workflow_that_does_nothing.run(start_event=StartEvent(query="llama"))
    )

    """
    ### Classify the Query with Structured Outputs
    """
    logger.info("### Classify the Query with Structured Outputs")

    class SaveToLlamaIndexDocs(BaseModel):
        """The URLs to parse and save into a llama-index specific docs collection."""

        llama_index_urls: List[str] = Field(default_factory=list)

    class SaveToWeaviateDocs(BaseModel):
        """The URLs to parse and save into a weaviate specific docs collection."""

        weaviate_urls: List[str] = Field(default_factory=list)

    class Ask(BaseModel):
        """The natural language questions that can be asked to a Q&A agent."""

        queries: List[str] = Field(default_factory=list)

    class Actions(BaseModel):
        """Actions to take based on the latest user message."""

        actions: List[
            Union[SaveToLlamaIndexDocs, SaveToWeaviateDocs, Ask]
        ] = Field(default_factory=list)

    """
    #### Create a Workflow
    
    Let's create a workflow that, still, does nothing, but the incoming user query will be converted to our structure. Based on the contents of that structure, the workflow will decide which step to run.
    
    Notice how whichever step runs first, will return a `StopEvent`... This is good, but maybe we can improve that later!
    """
    logger.info("#### Create a Workflow")

    class DocsAssistantWorkflow(Workflow):
        def __init__(self, *args, **kwargs):
            self.llm = OllamaFunctionCallingAdapterResponses(model="llama3.2")
            self.system_prompt = """You are a docs assistant. You evaluate incoming queries and break them down to subqueries when needed.
                              You decide on the next best course of action. Overall, here are the options:
                              - You can write the contents of a URL to llamaindex docs (if it's a llamaindex url)
                              - You can write the contents of a URL to weaviate docs (if it's a weaviate url)
                              - You can answer a question about llamaindex and weaviate using the QueryAgent"""
            super().__init__(*args, **kwargs)

        @step
        async def start(self, ev: StartEvent) -> EvaluateQuery:
            return EvaluateQuery(query=ev.query)

        @step
        async def evaluate_query(
            self, ev: EvaluateQuery
        ) -> QueryAgentEvent | WriteLlamaIndexDocsEvent | WriteWeaviateDocsEvent:
            sllm = self.llm.as_structured_llm(Actions)
            response = sllm.chat(
                [
                    ChatMessage(role="system", content=self.system_prompt),
                    ChatMessage(role="user", content=ev.query),
                ]
            )
            logger.success(format_json(response))
            actions = response.raw.actions
            logger.debug(actions)
            for action in actions:
                if isinstance(action, SaveToLlamaIndexDocs):
                    return WriteLlamaIndexDocsEvent(urls=action.llama_index_urls)
                elif isinstance(action, SaveToWeaviateDocs):
                    return WriteWeaviateDocsEvent(urls=action.weaviate_urls)
                elif isinstance(action, Ask):
                    for query in action.queries:
                        return QueryAgentEvent(query=query)

        @step
        async def write_li_docs(self, ev: WriteLlamaIndexDocsEvent) -> StopEvent:
            logger.debug(f"Writing {ev.urls} to LlamaIndex Docs")
            return StopEvent()

        @step
        async def write_weaviate_docs(
            self, ev: WriteWeaviateDocsEvent
        ) -> StopEvent:
            logger.debug(f"Writing {ev.urls} to Weaviate Docs")
            return StopEvent()

        @step
        async def query_agent(self, ev: QueryAgentEvent) -> StopEvent:
            logger.debug(f"Sending `'{ev.query}`' to agent")
            return StopEvent()

    everything_docs_agent_beta = DocsAssistantWorkflow()

    async def run_docs_agent_beta(query: str):
        logger.debug(
            await everything_docs_agent_beta.run(
                start_event=StartEvent(query=query)
            )
        )

    await run_docs_agent_beta(
        """Can you save https://www.llamaindex.ai/blog/get-citations-and-reasoning-for-extracted-data-in-llamaextract
        and https://www.llamaindex.ai/blog/llamaparse-update-may-2025-new-models-skew-detection-and-more??"""
    )

    await run_docs_agent_beta(
        "How many documents do we have in the LlamaIndexDocs collection now?"
    )

    await run_docs_agent_beta("What are LlamaIndex workflows?")

    await run_docs_agent_beta(
        "Can you save https://weaviate.io/blog/graph-rag and https://weaviate.io/blog/genai-apps-with-weaviate-and-databricks??"
    )

    """
    ## Run Multiple Branches & Put it all togehter
    
    In these cases, it makes sense to run multiple branches. So, a single step can trigger multiple events at once! We can `send_event` via the context 👇
    """
    logger.info("## Run Multiple Branches & Put it all togehter")

    class ActionCompleted(Event):
        result: str

    class DocsAssistantWorkflow(Workflow):
        def __init__(self, *args, **kwargs):
            self.llm = OllamaFunctionCallingAdapterResponses(model="llama3.2")
            self.system_prompt = """You are a docs assistant. You evaluate incoming queries and break them down to subqueries when needed.
                          You decide on the next best course of action. Overall, here are the options:
                          - You can write the contents of a URL to llamaindex docs (if it's a llamaindex url)
                          - You can write the contents of a URL to weaviate docs (if it's a weaviate url)
                          - You can answer a question about llamaindex and weaviate using the QueryAgent"""
            super().__init__(*args, **kwargs)

        @step
        async def start(self, ctx: Context, ev: StartEvent) -> EvaluateQuery:
            return EvaluateQuery(query=ev.query)

        @step
        async def evaluate_query(
            self, ctx: Context, ev: EvaluateQuery
        ) -> QueryAgentEvent | WriteLlamaIndexDocsEvent | WriteWeaviateDocsEvent | None:
            await ctx.store.set("results", [])
            sllm = self.llm.as_structured_llm(Actions)
            response = sllm.chat(
                [
                    ChatMessage(role="system", content=self.system_prompt),
                    ChatMessage(role="user", content=ev.query),
                ]
            )
            logger.success(format_json(response))
            actions = response.raw.actions
            await ctx.store.set("num_events", len(actions))
            await ctx.store.set("results", [])
            logger.debug(actions)
            for action in actions:
                if isinstance(action, SaveToLlamaIndexDocs):
                    ctx.send_event(
                        WriteLlamaIndexDocsEvent(urls=action.llama_index_urls)
                    )
                elif isinstance(action, SaveToWeaviateDocs):
                    ctx.send_event(
                        WriteWeaviateDocsEvent(urls=action.weaviate_urls)
                    )
                elif isinstance(action, Ask):
                    for query in action.queries:
                        ctx.send_event(QueryAgentEvent(query=query))

        @step
        async def write_li_docs(
            self, ctx: Context, ev: WriteLlamaIndexDocsEvent
        ) -> ActionCompleted:
            logger.debug(f"Writing {ev.urls} to LlamaIndex Docs")
            write_webpages_to_weaviate(
                client, urls=ev.urls, collection_name="LlamaIndexDocs"
            )
            results = await ctx.store.get("results")
            logger.success(format_json(results))
            results.append(f"Wrote {ev.urls} it LlamaIndex Docs")
            return ActionCompleted(result=f"Writing {ev.urls} to LlamaIndex Docs")

        @step
        async def write_weaviate_docs(
            self, ctx: Context, ev: WriteWeaviateDocsEvent
        ) -> ActionCompleted:
            logger.debug(f"Writing {ev.urls} to Weaviate Docs")
            write_webpages_to_weaviate(
                client, urls=ev.urls, collection_name="WeaviateDocs"
            )
            results = await ctx.store.get("results")
            logger.success(format_json(results))
            results.append(f"Wrote {ev.urls} it Weavite Docs")
            return ActionCompleted(result=f"Writing {ev.urls} to Weaviate Docs")

        @step
        async def query_agent(
            self, ctx: Context, ev: QueryAgentEvent
        ) -> ActionCompleted:
            logger.debug(f"Sending {ev.query} to agent")
            response = weaviate_agent.run(ev.query)
            results = await ctx.store.get("results")
            logger.success(format_json(results))
            results.append(
                f"QueryAgent responded with:\n {response.final_answer}")
            return ActionCompleted(result=f"Sending `'{ev.query}`' to agent")

        @step
        async def collect(
            self, ctx: Context, ev: ActionCompleted
        ) -> StopEvent | None:
            num_events = await ctx.store.get("num_events")
            logger.success(format_json(num_events))
            evs = ctx.collect_events(ev, [ActionCompleted] * num_events)
            if evs is None:
                return None
            return StopEvent(result=[ev.result for ev in evs])

    everything_docs_agent = DocsAssistantWorkflow(timeout=None)

    async def run_docs_agent(query: str):
        handler = everything_docs_agent.run(
            start_event=StartEvent(query=query))
        result = await handler
        logger.success(format_json(result))
        for response in await handler.ctx.store.get("results"):
            logger.debug(response)

    await run_docs_agent(
        "Can you save https://docs.llamaindex.ai/en/stable/understanding/workflows/ and https://docs.llamaindex.ai/en/stable/understanding/workflows/branches_and_loops/"
    )

    await run_docs_agent(
        "How many documents do we have in the LlamaIndexDocs collection now?"
    )

    await run_docs_agent(
        "What are LlamaIndex workflows? And can you save https://weaviate.io/blog/graph-rag"
    )

    await run_docs_agent("How do I use loops in llamaindex workflows?")

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
