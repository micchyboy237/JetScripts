async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.llms import ChatMessage
    from llama_index.core.memory import (
    VectorMemory,
    SimpleComposableMemory,
    ChatMemoryBuffer,
    )
    from llama_index.core.tools import FunctionTool
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/memory/composable_memory.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Simple Composable Memory
    
    **NOTE:** This example of memory is deprecated in favor of the newer and more flexible `Memory` class. See the [latest docs](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/).
    
    In this notebook, we demonstrate how to inject multiple memory sources into an agent. Specifically, we use the `SimpleComposableMemory` which is comprised of a `primary_memory` as well as potentially several secondary memory sources (stored in `secondary_memory_sources`). The main difference is that `primary_memory` will be used as the main chat buffer for the agent, where as any retrieved messages from `secondary_memory_sources` will be injected to the system prompt message only.
    
    Multiple memory sources may be of use for example in situations where you have a longer-term memory such as `VectorMemory` that you want to use in addition to the default `ChatMemoryBuffer`. What you'll see in this notebook is that with a `SimpleComposableMemory` you'll be able to effectively "load" the desired messages from long-term memory into the main memory (i.e. the `ChatMemoryBuffer`).
    
    ## How `SimpleComposableMemory` Works?
    
    We begin with the basic usage of the `SimpleComposableMemory`. Here we construct a `VectorMemory` as well as a default `ChatMemoryBuffer`. The `VectorMemory` will be our secondary memory source, whereas the `ChatMemoryBuffer` will be the main or primary one. To instantiate a `SimpleComposableMemory` object, we need to supply a `primary_memory` and (optionally) a list of `secondary_memory_sources`.
    
    ![SimpleComposableMemoryIllustration](https://d3ddy8balm3goa.cloudfront.net/llamaindex/simple-composable-memory.excalidraw.svg)
    """
    logger.info("# Simple Composable Memory")
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    
    
    vector_memory = VectorMemory.from_defaults(
        vector_store=None,  # leave as None to use default in-memory vector store
        embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
        retriever_kwargs={"similarity_top_k": 1},
    )
    
    msgs = [
        ChatMessage.from_str("You are a SOMEWHAT helpful assistant.", "system"),
        ChatMessage.from_str("Bob likes burgers.", "user"),
        ChatMessage.from_str("Indeed, Bob likes apples.", "assistant"),
        ChatMessage.from_str("Alice likes apples.", "user"),
    ]
    vector_memory.set(msgs)
    
    chat_memory_buffer = ChatMemoryBuffer.from_defaults()
    
    composable_memory = SimpleComposableMemory.from_defaults(
        primary_memory=chat_memory_buffer,
        secondary_memory_sources=[vector_memory],
    )
    
    composable_memory.primary_memory
    
    composable_memory.secondary_memory_sources
    
    """
    ### `put()` messages into memory
    
    Since `SimpleComposableMemory` is itself a subclass of `BaseMemory`, we add messages to it in the same way as we do for other memory modules. Note that for `SimpleComposableMemory`, invoking `.put()` effectively calls `.put()` on all memory sources. In other words, the message gets added to `primary` and `secondary` sources.
    """
    logger.info("### `put()` messages into memory")
    
    msgs = [
        ChatMessage.from_str("You are a REALLY helpful assistant.", "system"),
        ChatMessage.from_str("Jerry likes juice.", "user"),
    ]
    
    for m in msgs:
        composable_memory.put(m)
    
    """
    ### `get()` messages from memory
    
    When `.get()` is invoked, we similarly execute all of the `.get()` methods of `primary` memory as well as all of the `secondary` sources. This leaves us with sequence of lists of messages that we have to must "compose" into a sensible single set of messages (to pass downstream to our agents). Special care must be applied here in general to ensure that the final sequence of messages are both sensible and conform to the chat APIs of the LLM provider.
    
    For `SimpleComposableMemory`, we **inject the messages from the `secondary` sources in the system message of the `primary` memory**. The rest of the message history of the `primary` source is left intact, and this composition is what is ultimately returned.
    """
    logger.info("### `get()` messages from memory")
    
    msgs = composable_memory.get("What does Bob like?")
    msgs
    
    logger.debug(msgs[0])
    
    """
    ### Successive calls to `get()`
    
    Successive calls of `get()` will simply replace the loaded `secondary` memory messages in the system prompt.
    """
    logger.info("### Successive calls to `get()`")
    
    msgs = composable_memory.get("What does Alice like?")
    msgs
    
    logger.debug(msgs[0])
    
    """
    ### What if `get()` retrieves `secondary` messages that already exist in `primary` memory?
    
    In the event that messages retrieved from `secondary` memory already exist in `primary` memory, then these rather redundant secondary messages will not get added to the system message. In the below example, the message "Jerry likes juice." was `put` into all memory sources, so the system message is not altered.
    """
    logger.info("### What if `get()` retrieves `secondary` messages that already exist in `primary` memory?")
    
    msgs = composable_memory.get("What does Jerry like?")
    msgs
    
    """
    ### How to `reset` memory
    
    Similar to the other methods `put()` and `get()`, calling `reset()` will execute `reset()` on both the `primary` and `secondary` memory sources. If you want to reset only the `primary` then you should call the `reset()` method only from it.
    
    #### `reset()` only primary memory
    """
    logger.info("### How to `reset` memory")
    
    composable_memory.primary_memory.reset()
    
    composable_memory.primary_memory.get()
    
    composable_memory.secondary_memory_sources[0].get("What does Alice like?")
    
    """
    #### `reset()` all memory sources
    """
    logger.info("#### `reset()` all memory sources")
    
    composable_memory.reset()
    
    composable_memory.primary_memory.get()
    
    """
    ## Use `SimpleComposableMemory` With An Agent
    
    Here we will use a `SimpleComposableMemory` with an agent and demonstrate how a secondary, long-term memory source can be used to use messages from on agent conversation as part of another conversation with another agent session.
    """
    logger.info("## Use `SimpleComposableMemory` With An Agent")
    
    
    """
    ### Define our memory modules
    """
    logger.info("### Define our memory modules")
    
    vector_memory = VectorMemory.from_defaults(
        vector_store=None,  # leave as None to use default in-memory vector store
        embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
        retriever_kwargs={"similarity_top_k": 2},
    )
    
    chat_memory_buffer = ChatMemoryBuffer.from_defaults()
    
    composable_memory = SimpleComposableMemory.from_defaults(
        primary_memory=chat_memory_buffer,
        secondary_memory_sources=[vector_memory],
    )
    
    """
    ### Define our Agent
    """
    logger.info("### Define our Agent")
    
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and returns the result integer"""
        return a * b
    
    
    def mystery(a: int, b: int) -> int:
        """Mystery function on two numbers"""
        return a**2 - b**2
    
    
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    mystery_tool = FunctionTool.from_defaults(fn=mystery)
    
    llm = OllamaFunctionCallingAdapter(model="llama3.2", log_dir=f"{LOG_DIR}/chats")
    agent = FunctionAgent(
        tools=[multiply_tool, mystery_tool],
        llm=llm,
    )
    
    """
    ### Execute some function calls
    
    When `.chat()` is invoked, the messages are put into the composable memory, which we understand from the previous section implies that all the messages are put in both `primary` and `secondary` sources.
    """
    logger.info("### Execute some function calls")
    
    response = await agent.run(
            "What is the mystery function on 5 and 6?", memory=composable_memory
        )
    logger.success(format_json(response))
    logger.debug(str(response))
    
    response = await agent.run(
            "What happens if you multiply 2 and 3?", memory=composable_memory
        )
    logger.success(format_json(response))
    logger.debug(str(response))
    
    """
    ### New Agent Sessions
    
    Now that we've added the messages to our `vector_memory`, we can see the effect of having this memory be used with a new agent session versus when it is used. Specifically, we ask the new agents to "recall" the outputs of the function calls, rather than re-computing.
    
    #### An Agent without our past memory
    """
    logger.info("### New Agent Sessions")
    
    response = await agent.run(
            "What was the output of the mystery function on 5 and 6 again? Don't recompute."
        )
    logger.success(format_json(response))
    
    logger.debug(str(response))
    
    """
    #### An Agent with our past memory
    
    We see that the agent without access to the our past memory cannot complete the task. With this next agent we will indeed pass in our previous long-term memory (i.e., `vector_memory`). Note that we even use a fresh `ChatMemoryBuffer` which means there is no `chat_history` with this agent. Nonetheless, it will be able to retrieve from our long-term memory to get the past dialogue it needs.
    """
    logger.info("#### An Agent with our past memory")
    
    response = await agent.run(
            "What was the output of the mystery function on 5 and 6 again? Don't recompute.",
            memory=composable_memory,
        )
    logger.success(format_json(response))
    logger.debug(str(response))
    
    response = await agent.run(
            "What was the output of the multiply function on 2 and 3 again? Don't recompute.",
            memory=composable_memory,
        )
    logger.success(format_json(response))
    logger.debug(str(response))
    
    """
    ### What happens under the hood with `.run(user_input)`
    
    Under the hood, `.run(user_input)` call effectively will call the memory's `.get()` method with `user_input` as the argument. As we learned in the previous section, this will ultimately return a composition of the `primary` and all of the `secondary` memory sources. These composed messages are what is being passed to the LLM's chat API as the chat history.
    """
    logger.info("### What happens under the hood with `.run(user_input)`")
    
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