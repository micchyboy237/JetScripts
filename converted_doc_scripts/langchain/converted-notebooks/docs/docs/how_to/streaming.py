from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import (
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables import chain
    from langchain_core.tools import tool
    import ChatModelTabs from "@theme/ChatModelTabs";
    import langchain_core
    import os
    import shutil

    async def main():
    JsonOutputParser,
)

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
    keywords: [stream]
    ---

    # How to stream runnables

    :::info Prerequisites

    This guide assumes familiarity with the following concepts:
    - [Chat models](/docs/concepts/chat_models)
    - [LangChain Expression Language](/docs/concepts/lcel)
    - [Output parsers](/docs/concepts/output_parsers)

    :::

    Streaming is critical in making applications based on LLMs feel responsive to end-users.

    Important LangChain primitives like [chat models](/docs/concepts/chat_models), [output parsers](/docs/concepts/output_parsers), [prompts](/docs/concepts/prompt_templates), [retrievers](/docs/concepts/retrievers), and [agents](/docs/concepts/agents) implement the LangChain [Runnable Interface](/docs/concepts/runnables).

    This interface provides two general approaches to stream content:

    1. sync `stream` and async `astream`: a **default implementation** of streaming that streams the **final output** from the chain.
    2. async `astream_events` and async `astream_log`: these provide a way to stream both **intermediate steps** and **final output** from the chain.

    Let's take a look at both approaches, and try to understand how to use them.

    :::info
    For a higher-level overview of streaming techniques in LangChain, see [this section of the conceptual guide](/docs/concepts/streaming).
    :::

    ## Using Stream

    All `Runnable` objects implement a sync method called `stream` and an async variant called `astream`.

    These methods are designed to stream the final output in chunks, yielding each chunk as soon as it is available.

    Streaming is only possible if all steps in the program know how to process an **input stream**; i.e., process an input chunk one at a time, and yield a corresponding output chunk.

    The complexity of this processing can vary, from straightforward tasks like emitting tokens produced by an LLM, to more challenging ones like streaming parts of JSON results before the entire JSON is complete.

    The best place to start exploring streaming is with the single most important components in LLMs apps-- the LLMs themselves!

    ### LLMs and Chat Models

    Large language models and their chat variants are the primary bottleneck in LLM based apps.

    Large language models can take **several seconds** to generate a complete response to a query. This is far slower than the **~200-300 ms** threshold at which an application feels responsive to an end user.

    The key strategy to make the application feel more responsive is to show intermediate progress; viz., to stream the output from the model **token by token**.

    We will show examples of streaming using a chat model. Choose one from the options below:


    <ChatModelTabs
      customVarName="model"
    />
    """
 logger.info("# How to stream runnables")

  # from getpass import getpass

  keys = [
       #     "ANTHROPIC_API_KEY",
        #     "OPENAI_API_KEY",
       ]

   for key in keys:
        if key not in os.environ:
           #         os.environ[key] = getpass(f"Enter API Key for {key}=?")

    model = ChatOllama(model="llama3.2")

    """
    Let's start with the sync `stream` API:
    """
    logger.info("Let's start with the sync `stream` API:")

    chunks = []
    for chunk in model.stream("what color is the sky?"):
        chunks.append(chunk)
        logger.debug(chunk.content, end="|", flush=True)

    """
    Alternatively, if you're working in an async environment, you may consider using the async `astream` API:
    """
    logger.info(
        "Alternatively, if you're working in an async environment, you may consider using the async `astream` API:")

    chunks = []
    for chunk in model.stream("what color is the sky?"):
        chunks.append(chunk)
        logger.debug(chunk.content, end="|", flush=True)

    """
    Let's inspect one of the chunks
    """
    logger.info("Let's inspect one of the chunks")

    chunks[0]

    """
    We got back something called an `AIMessageChunk`. This chunk represents a part of an `AIMessage`.
    
    Message chunks are additive by design -- one can simply add them up to get the state of the response so far!
    """
    logger.info(
        "We got back something called an `AIMessageChunk`. This chunk represents a part of an `AIMessage`.")

    chunks[0] + chunks[1] + chunks[2] + chunks[3] + chunks[4]

    """
    ### Chains
    
    Virtually all LLM applications involve more steps than just a call to a language model.
    
    Let's build a simple chain using `LangChain Expression Language` (`LCEL`) that combines a prompt, model and a parser and verify that streaming works.
    
    We will use [`StrOutputParser`](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html) to parse the output from the model. This is a simple parser that extracts the `content` field from an `AIMessageChunk`, giving us the `token` returned by the model.
    
    :::tip
    LCEL is a *declarative* way to specify a "program" by chainining together different LangChain primitives. Chains created using LCEL benefit from an automatic implementation of `stream` and `astream` allowing streaming of the final output. In fact, chains created with LCEL implement the entire standard Runnable interface.
    :::
    """
    logger.info("### Chains")

    prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
    parser = StrOutputParser()
    chain = prompt | model | parser

    for chunk in chain.stream({"topic": "parrot"}):
        logger.debug(chunk, end="|", flush=True)

    """
    Note that we're getting streaming output even though we're using `parser` at the end of the chain above. The `parser` operates on each streaming chunk individidually. Many of the [LCEL primitives](/docs/how_to#langchain-expression-language-lcel) also support this kind of transform-style passthrough streaming, which can be very convenient when constructing apps. 
    
    Custom functions can be [designed to return generators](/docs/how_to/functions#streaming), which are able to operate on streams.
    
    Certain runnables, like [prompt templates](/docs/how_to#prompt-templates) and [chat models](/docs/how_to#chat-models), cannot process individual chunks and instead aggregate all previous steps. Such runnables can interrupt the streaming process.
    
    :::note
    The LangChain Expression language allows you to separate the construction of a chain from the mode in which it is used (e.g., sync/async, batch/streaming etc.). If this is not relevant to what you're building, you can also rely on a standard **imperative** programming approach by
    caling `invoke`, `batch` or `stream` on each component individually, assigning the results to variables and then using them downstream as you see fit.
    
    :::
    
    ### Working with Input Streams
    
    What if you wanted to stream JSON from the output as it was being generated?
    
    If you were to rely on `json.loads` to parse the partial json, the parsing would fail as the partial json wouldn't be valid json.
    
    You'd likely be at a complete loss of what to do and claim that it wasn't possible to stream JSON.
    
    Well, turns out there is a way to do it -- the parser needs to operate on the **input stream**, and attempt to "auto-complete" the partial json into a valid state.
    
    Let's see such a parser in action to understand what this means.
    """
    logger.info("### Working with Input Streams")

    chain = (
        model | JsonOutputParser()
    )  # Due to a bug in older versions of Langchain, JsonOutputParser did not stream results from some models
    for text in chain.stream(
        "output a list of the countries france, spain and japan and their populations in JSON format. "
        'Use a dict with an outer key of "countries" which contains a list of countries. '
        "Each country should have the key `name` and `population`"
    ):
        logger.debug(text, flush=True)

    """
    Now, let's **break** streaming. We'll use the previous example and append an extraction function at the end that extracts the country names from the finalized JSON.
    
    :::warning
    Any steps in the chain that operate on **finalized inputs** rather than on **input streams** can break streaming functionality via `stream` or `astream`.
    :::
    
    :::tip
    Later, we will discuss the `astream_events` API which streams results from intermediate steps. This API will stream results from intermediate steps even if the chain contains steps that only operate on **finalized inputs**.
    :::
    """
    logger.info("Now, let's **break** streaming. We'll use the previous example and append an extraction function at the end that extracts the country names from the finalized JSON.")

    def _extract_country_names(inputs):
        """A function that does not operates on input streams and breaks streaming."""
        if not isinstance(inputs, dict):
            return ""

        if "countries" not in inputs:
            return ""

        countries = inputs["countries"]

        if not isinstance(countries, list):
            return ""

        country_names = [
            country.get("name") for country in countries if isinstance(country, dict)
        ]
        return country_names

    chain = model | JsonOutputParser() | _extract_country_names

    for text in chain.stream(
        "output a list of the countries france, spain and japan and their populations in JSON format. "
        'Use a dict with an outer key of "countries" which contains a list of countries. '
        "Each country should have the key `name` and `population`"
    ):
        logger.debug(text, end="|", flush=True)

    """
    #### Generator Functions
    
    Let's fix the streaming using a generator function that can operate on the **input stream**.
    
    :::tip
    A generator function (a function that uses `yield`) allows writing code that operates on **input streams**
    :::
    """
    logger.info("#### Generator Functions")

    async def _extract_country_names_streaming(input_stream):
        """A function that operates on input streams."""
        country_names_so_far = set()

        async for input in input_stream:
            if not isinstance(input, dict):
                continue

            if "countries" not in input:
                continue

            countries = input["countries"]

            if not isinstance(countries, list):
                continue

            for country in countries:
                name = country.get("name")
                if not name:
                    continue
                if name not in country_names_so_far:
                    yield name
                    country_names_so_far.add(name)

    chain = model | JsonOutputParser() | _extract_country_names_streaming

    for text in chain.stream(
        "output a list of the countries france, spain and japan and their populations in JSON format. "
        'Use a dict with an outer key of "countries" which contains a list of countries. '
        "Each country should have the key `name` and `population`",
    ):
        logger.debug(text, end="|", flush=True)

    """
    :::note
    Because the code above is relying on JSON auto-completion, you may see partial names of countries (e.g., `Sp` and `Spain`), which is not what one would want for an extraction result!
    
    We're focusing on streaming concepts, not necessarily the results of the chains.
    :::
    
    ### Non-streaming components
    
    Some built-in components like Retrievers do not offer any `streaming`. What happens if we try to `stream` them? 🤨
    """
    logger.info("### Non-streaming components")

    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    vectorstore = FAISS.from_texts(
        ["harrison worked at kensho", "harrison likes spicy food"],
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
    )
    retriever = vectorstore.as_retriever()

    chunks = [chunk for chunk in retriever.stream("where did harrison work?")]
    chunks

    """
    Stream just yielded the final result from that component.
    
    This is OK 🥹! Not all components have to implement streaming -- in some cases streaming is either unnecessary, difficult or just doesn't make sense.
    
    :::tip
    An LCEL chain constructed using non-streaming components, will still be able to stream in a lot of cases, with streaming of partial output starting after the last non-streaming step in the chain.
    :::
    """
    logger.info("Stream just yielded the final result from that component.")

    retrieval_chain = (
        {
            "context": retriever.with_config(run_name="Docs"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    for chunk in retrieval_chain.stream(
        "Where did harrison work? Write 3 made up sentences about this place."
    ):
        logger.debug(chunk, end="|", flush=True)

    """
    Now that we've seen how `stream` and `astream` work, let's venture into the world of streaming events. 🏞️
    
    ## Using Stream Events
    
    Event Streaming is a **beta** API. This API may change a bit based on feedback.
    
    :::note
    
    This guide demonstrates the `V2` API and requires langchain-core >= 0.2. For the `V1` API compatible with older versions of LangChain, see [here](https://python.langchain.com/v0.1/docs/expression_language/streaming/#using-stream-events).
    :::
    """
    logger.info("## Using Stream Events")

    langchain_core.__version__

    """
    For the `astream_events` API to work properly:
    
    * Use `async` throughout the code to the extent possible (e.g., async tools etc)
    * Propagate callbacks if defining custom functions / runnables
    * Whenever using runnables without LCEL, make sure to call `.astream()` on LLMs rather than `.ainvoke` to force the LLM to stream tokens.
    * Let us know if anything doesn't work as expected! :)
    
    ### Event Reference
    
    Below is a reference table that shows some events that might be emitted by the various Runnable objects.
    
    
    :::note
    When streaming is implemented properly, the inputs to a runnable will not be known until after the input stream has been entirely consumed. This means that `inputs` will often be included only for `end` events and rather than for `start` events.
    :::
    
    | event                | name             | chunk                           | input                                         | output                                          |
    |----------------------|------------------|---------------------------------|-----------------------------------------------|-------------------------------------------------|
    | on_chat_model_start  | [model name]     |                                 | \{"messages": [[SystemMessage, HumanMessage]]\} |                                                 |
    | on_chat_model_stream | [model name]     | AIMessageChunk(content="hello") |                                               |                                                 |
    | on_chat_model_end    | [model name]     |                                 | \{"messages": [[SystemMessage, HumanMessage]]\} | AIMessageChunk(content="hello world")           |
    | on_llm_start         | [model name]     |                                 | \{'input': 'hello'\}                            |                                                 |
    | on_llm_stream        | [model name]     | 'Hello'                         |                                               |                                                 |
    | on_llm_end           | [model name]     |                                 | 'Hello human!'                                |                                                 |
    | on_chain_start       | format_docs      |                                 |                                               |                                                 |
    | on_chain_stream      | format_docs      | "hello world!, goodbye world!"  |                                               |                                                 |
    | on_chain_end         | format_docs      |                                 | [Document(...)]                               | "hello world!, goodbye world!"                  |
    | on_tool_start        | some_tool        |                                 | \{"x": 1, "y": "2"\}                            |                                                 |
    | on_tool_end          | some_tool        |                                 |                                               | \{"x": 1, "y": "2"\}                              |
    | on_retriever_start   | [retriever name] |                                 | \{"query": "hello"\}                            |                                                 |
    | on_retriever_end     | [retriever name] |                                 | \{"query": "hello"\}                            | [Document(...), ..]                             |
    | on_prompt_start      | [template_name]  |                                 | \{"question": "hello"\}                         |                                                 |
    | on_prompt_end        | [template_name]  |                                 | \{"question": "hello"\}                         | ChatPromptValue(messages: [SystemMessage, ...]) |
    
    ### Chat Model
    
    Let's start off by looking at the events produced by a chat model.
    """
    logger.info("### Event Reference")

    events = []
    for event in model.stream_events("hello"):
        events.append(event)

    """
    :::note
    
    For `langchain-core<0.3.37`, set the `version` kwarg explicitly (e.g., `model.astream_events("hello", version="v2")`).
    
    :::
    
    Let's take a look at the few of the start event and a few of the end events.
    """
    logger.info("For `langchain-core<0.3.37`, set the `version` kwarg explicitly (e.g., `model.astream_events("hello", version="v2")`).")

    events[:3]

    events[-2:]

    """
    ### Chain
    
    Let's revisit the example chain that parsed streaming JSON to explore the streaming events API.
    """
    logger.info("### Chain")

    chain = (
        model | JsonOutputParser()
    )  # Due to a bug in older versions of Langchain, JsonOutputParser did not stream results from some models

    events = [
        event
        for event in chain.stream_events(
            "output a list of the countries france, spain and japan and their populations in JSON format. "
            'Use a dict with an outer key of "countries" which contains a list of countries. '
            "Each country should have the key `name` and `population`",
        )
    ]

    """
    If you examine at the first few events, you'll notice that there are **3** different start events rather than **2** start events.
    
    The three start events correspond to:
    
    1. The chain (model + parser)
    2. The model
    3. The parser
    """
    logger.info("If you examine at the first few events, you'll notice that there are **3** different start events rather than **2** start events.")

    events[:3]

    """
    What do you think you'd see if you looked at the last 3 events? what about the middle?
    
    Let's use this API to take output the stream events from the model and the parser. We're ignoring start events, end events and events from the chain.
    """
    logger.info(
        "What do you think you'd see if you looked at the last 3 events? what about the middle?")

    num_events = 0

    for event in chain.stream_events(
        "output a list of the countries france, spain and japan and their populations in JSON format. "
        'Use a dict with an outer key of "countries" which contains a list of countries. '
        "Each country should have the key `name` and `population`",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            logger.debug(
                f"Chat model chunk: {repr(event['data']['chunk'].content)}",
                flush=True,
            )
        if kind == "on_parser_stream":
            logger.debug(f"Parser chunk: {event['data']['chunk']}", flush=True)
        num_events += 1
        if num_events > 30:
            logger.debug("...")
            break

    """
    Because both the model and the parser support streaming, we see streaming events from both components in real time! Kind of cool isn't it? 🦜
    
    ### Filtering Events
    
    Because this API produces so many events, it is useful to be able to filter on events.
    
    You can filter by either component `name`, component `tags` or component `type`.
    
    #### By Name
    """
    logger.info("### Filtering Events")

    chain = model.with_config({"run_name": "model"}) | JsonOutputParser().with_config(
        {"run_name": "my_parser"}
    )

    max_events = 0
    for event in chain.stream_events(
        "output a list of the countries france, spain and japan and their populations in JSON format. "
        'Use a dict with an outer key of "countries" which contains a list of countries. '
        "Each country should have the key `name` and `population`",
        include_names=["my_parser"],
    ):
        logger.debug(event)
        max_events += 1
        if max_events > 10:
            logger.debug("...")
            break

    """
    #### By Type
    """
    logger.info("#### By Type")

    chain = model.with_config({"run_name": "model"}) | JsonOutputParser().with_config(
        {"run_name": "my_parser"}
    )

    max_events = 0
    for event in chain.stream_events(
        'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
        include_types=["chat_model"],
    ):
        logger.debug(event)
        max_events += 1
        if max_events > 10:
            logger.debug("...")
            break

    """
    #### By Tags
    
    :::caution
    
    Tags are inherited by child components of a given runnable. 
    
    If you're using tags to filter, make sure that this is what you want.
    :::
    """
    logger.info("#### By Tags")

    chain = (model | JsonOutputParser()).with_config({"tags": ["my_chain"]})

    max_events = 0
    for event in chain.stream_events(
        'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
        include_tags=["my_chain"],
    ):
        logger.debug(event)
        max_events += 1
        if max_events > 10:
            logger.debug("...")
            break

    """
    ### Non-streaming components
    
    Remember how some components don't stream well because they don't operate on **input streams**?
    
    While such components can break streaming of the final output when using `astream`, `astream_events` will still yield streaming events from intermediate steps that support streaming!
    """
    logger.info("### Non-streaming components")

    def _extract_country_names(inputs):
        """A function that does not operates on input streams and breaks streaming."""
        if not isinstance(inputs, dict):
            return ""

        if "countries" not in inputs:
            return ""

        countries = inputs["countries"]

        if not isinstance(countries, list):
            return ""

        country_names = [
            country.get("name") for country in countries if isinstance(country, dict)
        ]
        return country_names

    chain = (
        model | JsonOutputParser() | _extract_country_names
    )  # This parser only works with Ollama right now

    """
    As expected, the `astream` API doesn't work correctly because `_extract_country_names` doesn't operate on streams.
    """
    logger.info(
        "As expected, the `astream` API doesn't work correctly because `_extract_country_names` doesn't operate on streams.")

    for chunk in chain.stream(
        "output a list of the countries france, spain and japan and their populations in JSON format. "
        'Use a dict with an outer key of "countries" which contains a list of countries. '
        "Each country should have the key `name` and `population`",
    ):
        logger.debug(chunk, flush=True)

    """
    Now, let's confirm that with astream_events we're still seeing streaming output from the model and the parser.
    """
    logger.info(
        "Now, let's confirm that with astream_events we're still seeing streaming output from the model and the parser.")

    num_events = 0

    for event in chain.stream_events(
        "output a list of the countries france, spain and japan and their populations in JSON format. "
        'Use a dict with an outer key of "countries" which contains a list of countries. '
        "Each country should have the key `name` and `population`",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            logger.debug(
                f"Chat model chunk: {repr(event['data']['chunk'].content)}",
                flush=True,
            )
        if kind == "on_parser_stream":
            logger.debug(f"Parser chunk: {event['data']['chunk']}", flush=True)
        num_events += 1
        if num_events > 30:
            logger.debug("...")
            break

    """
    ### Propagating Callbacks
    
    :::caution
    If you're using invoking runnables inside your tools, you need to propagate callbacks to the runnable; otherwise, no stream events will be generated.
    :::
    
    :::note
    When using `RunnableLambdas` or `@chain` decorator, callbacks are propagated automatically behind the scenes.
    :::
    """
    logger.info("### Propagating Callbacks")

    def reverse_word(word: str):
        return word[::-1]

    reverse_word = RunnableLambda(reverse_word)

    @tool
    def bad_tool(word: str):
        """Custom tool that doesn't propagate callbacks."""
        return reverse_word.invoke(word)

    for event in bad_tool.stream_events("hello"):
        logger.debug(event)

    """
    Here's a re-implementation that does propagate callbacks correctly. You'll notice that now we're getting events from the `reverse_word` runnable as well.
    """
    logger.info("Here's a re-implementation that does propagate callbacks correctly. You'll notice that now we're getting events from the `reverse_word` runnable as well.")

    @tool
    def correct_tool(word: str, callbacks):
        """A tool that correctly propagates callbacks."""
        return reverse_word.invoke(word, {"callbacks": callbacks})

    for event in correct_tool.stream_events("hello"):
        logger.debug(event)

    """
    If you're invoking runnables from within Runnable Lambdas or `@chains`, then callbacks will be passed automatically on your behalf.
    """
    logger.info("If you're invoking runnables from within Runnable Lambdas or `@chains`, then callbacks will be passed automatically on your behalf.")

    async def reverse_and_double(word: str):
        return await reverse_word.ainvoke(word) * 2

    reverse_and_double = RunnableLambda(reverse_and_double)

    await reverse_and_double.ainvoke("1234")

    for event in reverse_and_double.stream_events("1234"):
        logger.debug(event)

    """
    And with the `@chain` decorator:
    """
    logger.info("And with the `@chain` decorator:")

    @chain
    async def reverse_and_double(word: str):
        return await reverse_word.ainvoke(word) * 2

    await reverse_and_double.ainvoke("1234")

    for event in reverse_and_double.stream_events("1234"):
        logger.debug(event)

    """
    ## Next steps
    
    Now you've learned some ways to stream both final outputs and internal steps with LangChain.
    
    To learn more, check out the other how-to guides in this section, or the [conceptual guide on Langchain Expression Language](/docs/concepts/lcel/).
    """
    logger.info("## Next steps")

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
